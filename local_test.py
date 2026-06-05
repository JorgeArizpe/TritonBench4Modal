from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from code_utils import _extract_code, _static_validate_code


SYSTEM_PROMPT = (
    "Generate only the requested code. The code must be a complete Python module "
    "with the necessary imports, Triton kernel definition(s), and the exact "
    "wrapper function definition. Do not include comments, notes, markdown, "
    "examples, tests, explanations, or disclaimers."
)

USER_PROMPT = (
    "You are an expert in Triton programming, capable of writing corresponding "
    "Triton kernels and wrapper functions based on functional descriptions and "
    "function parameters. Ensure that the wrapper function fully corresponds to "
    "the provided function information.\n"
    "Functional Description: Performs a fused operation combining batch matrix "
    "multiplication, RMS normalization, GELU activation, dropout, and "
    "subtraction. The function takes three input tensors, performs batch matrix "
    "multiplication on the first two, applies RMS normalization, GELU activation, "
    "and dropout, and finally subtracts the third tensor from the result.\n"
    "Wrapper Entry Information: "
    "fused_bmm_rmsnorm_gelu_dropout_sub(input1, input2, other, "
    "normalized_shape, dropout_p=0.5, training=True, approximate='none', "
    "eps=1e-5, *, out=None) -> Tensor. Args: input1 (Tensor): First input "
    "tensor for batch matrix multiplication, of shape (B, N, M), where B is the "
    "batch size. input2 (Tensor): Second input tensor for batch matrix "
    "multiplication, of shape (B, M, P). other (Tensor): Tensor to subtract "
    "from the result after dropout, must be broadcastable to the shape of the "
    "output. normalized_shape (int or list or torch.Size): Shape over which RMS "
    "normalization is applied, typically the size of the last dimension P. "
    "dropout_p (float, optional): Probability of an element to be zeroed in the "
    "dropout layer. Default: 0.5. training (bool, optional): Apply dropout if "
    "True. Default: True. approximate (str, optional): Can be 'none' or "
    "'tanh'. The approximation to use for GELU. Default: 'none'. eps (float, "
    "optional): A value added to the denominator for numerical stability in RMS "
    "normalization. Default: 1e-5. out (Tensor, optional): Output tensor. "
    "Ignored if None. Default: None. Shape: - Input1: (B, N, M), Input2: "
    "(B, M, P), Other: broadcastable to (B, N, P). Output: (B, N, P).\n"
    "After generation, verify if the Triton wrapper aligns with the provided "
    "func_inputs. If not, regenerate."
)

# Ollama's public Python API exposes constrained decoding through `format`.
# That constrains the response envelope token-by-token on the Ollama side.
TRITON_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "python_code": {
            "type": "string",
            "minLength": 200,
            "pattern": (
                "^[\\s\\S]*import\\s+torch[\\s\\S]*"
                "import\\s+triton[\\s\\S]*"
                "import\\s+triton\\.language\\s+as\\s+tl[\\s\\S]*"
                "@triton\\.jit[\\s\\S]*"
                "def\\s+fused_bmm_rmsnorm_gelu_dropout_sub\\s*\\([\\s\\S]*$"
            ),
            "description": (
                "A complete Python module. It must import torch, triton, and "
                "triton.language as tl; define at least one @triton.jit kernel; "
                "and define fused_bmm_rmsnorm_gelu_dropout_sub with the exact "
                "requested wrapper signature."
            ),
        }
    },
    "required": ["python_code"],
    "additionalProperties": False,
}

# Some Ollama builds/models cannot build a sampler from string `pattern`
# constraints and fail with "unable to create sampling context". This fallback
# still constrains the response envelope token-by-token as JSON with exactly one
# field; Triton/Python shape is checked immediately after generation.
OLLAMA_COMPAT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "python_code": {
            "type": "string",
            "description": (
                "A complete Python module containing Triton kernels and the "
                "requested wrapper function."
            ),
        }
    },
    "required": ["python_code"],
    "additionalProperties": False,
}

# True XGrammar mode uses this grammar as a logits mask. It intentionally keeps
# expression lines permissive while forcing the top-level Triton module shape:
# imports first, at least one @triton.jit function, then Python/Triton defs.
TRITON_MODULE_EBNF = r'''
root ::= module
module ::= import_section blank_line? jit_function blank_line? definition*
import_section ::= "import torch" newline "import triton" newline "import triton.language as tl" newline extra_import*
extra_import ::= ("import " text_no_newline newline) | ("from " text_no_newline newline)
definition ::= jit_function blank_line? | python_function blank_line?
jit_function ::= "@triton.jit" newline python_function
python_function ::= "def " identifier "(" parameter_text ")" ":" suite
suite ::= newline indented_line+
indented_line ::= indent statement? newline
statement ::= text_no_newline
parameter_text ::= [^)\n]*
text_no_newline ::= [^\n]+
identifier ::= [A-Za-z_] [A-Za-z0-9_]*
blank_line ::= newline newline*
indent ::= "    " | "\t"
newline ::= "\n"
'''


def _messages(expect_json: bool) -> list[dict[str, str]]:
    if expect_json:
        system = (
            SYSTEM_PROMPT
            + " Respond as JSON with exactly one key, python_code. The value "
            "must be raw code, not a markdown code block."
        )
    else:
        system = (
            SYSTEM_PROMPT
            + " Start the module with exactly these three import lines: "
            "import torch, import triton, import triton.language as tl."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT},
    ]


def _ollama_options(args: argparse.Namespace) -> dict[str, Any]:
    options: dict[str, Any] = {
        "num_predict": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.seed is not None:
        options["seed"] = args.seed
    return options


def generate_with_ollama(args: argparse.Namespace) -> str:
    try:
        from ollama import Client, ResponseError
    except ImportError as exc:
        raise RuntimeError(
            "The Ollama backend requires the Python package `ollama`. "
            "Install local dependencies with `pip install -r requirements.txt`."
        ) from exc

    client = Client(host=args.ollama_host) if args.ollama_host else Client()
    schema = (
        TRITON_OUTPUT_SCHEMA
        if args.ollama_schema == "strict"
        else OLLAMA_COMPAT_OUTPUT_SCHEMA
    )
    try:
        response = client.chat(
            model=args.model,
            messages=_messages(expect_json=True),
            format=schema,
            options=_ollama_options(args),
            stream=False,
        )
    except ResponseError as exc:
        can_retry_compat = (
            args.ollama_schema == "strict"
            and exc.status_code == 500
            and "sampling context" in exc.error.lower()
        )
        if not can_retry_compat:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc
        print(
            "warning: Ollama rejected the stricter schema sampler; retrying "
            "with the compatibility JSON schema.",
            file=sys.stderr,
        )
        try:
            response = client.chat(
                model=args.model,
                messages=_messages(expect_json=True),
                format=OLLAMA_COMPAT_OUTPUT_SCHEMA,
                options=_ollama_options(args),
                stream=False,
            )
        except ResponseError as retry_exc:
            raise RuntimeError(f"Ollama request failed: {retry_exc}") from retry_exc
    content = response["message"]["content"]
    if args.print_raw:
        print(content, file=sys.stderr)

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Ollama returned text that did not parse as the constrained JSON "
            "schema. If you used a small --max-tokens value, raise it so the "
            "model can close the JSON object; otherwise check that your Ollama "
            "version supports structured outputs."
        ) from exc

    code = payload.get("python_code")
    if not isinstance(code, str):
        raise RuntimeError("Constrained response did not contain python_code.")
    return _extract_code(code)


def _compile_triton_grammar(tokenizer: Any, config: Any) -> Any:
    try:
        import xgrammar as xgr
    except ImportError as exc:
        raise RuntimeError(
            "The xgrammar-hf backend requires `xgrammar`, `torch`, and "
            "`transformers`."
        ) from exc

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=config.vocab_size,
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    grammar = xgr.Grammar.from_ebnf(TRITON_MODULE_EBNF)
    try:
        return compiler.compile_grammar(grammar)
    except TypeError:
        # Older XGrammar builds accepted the EBNF string directly.
        return compiler.compile_grammar(TRITON_MODULE_EBNF)


def generate_with_xgrammar_hf(args: argparse.Namespace) -> str:
    try:
        import torch
        import xgrammar as xgr
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional true token-mask backend dependencies first: "
            "`pip install xgrammar torch transformers`."
        ) from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model,
        trust_remote_code=args.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        args.hf_model,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    messages = _messages(expect_json=False)
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in messages
        ) + "\nASSISTANT:"

    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    compiled_grammar = _compile_triton_grammar(tokenizer, config)
    logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)

    generate_kwargs: dict[str, Any] = {
        **model_inputs,
        "max_new_tokens": args.max_tokens,
        "logits_processor": [logits_processor],
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**generate_kwargs)

    new_tokens = generated_ids[0][model_inputs.input_ids.shape[-1] :]
    return _extract_code(tokenizer.decode(new_tokens, skip_special_tokens=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the local Triton test prompt with constrained decoding."
    )
    parser.add_argument(
        "--backend",
        choices=("ollama", "xgrammar-hf"),
        default=os.environ.get("LOCAL_TEST_BACKEND", "ollama"),
        help=(
            "ollama uses Ollama's server-side JSON-schema constraint; "
            "xgrammar-hf uses XGrammar logits masking with a Triton module grammar."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "Qwen3-Coder"),
        help="Ollama model name for --backend ollama.",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST"),
        help="Optional Ollama host URL, for example http://localhost:11434.",
    )
    parser.add_argument(
        "--ollama-schema",
        choices=("strict", "compat"),
        default=os.environ.get("OLLAMA_SCHEMA", "compat"),
        help=(
            "strict adds schema pattern checks for Triton imports/wrapper; "
            "compat uses a simpler schema for Ollama builds that reject patterns."
        ),
    )
    parser.add_argument(
        "--hf-model",
        default=os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        help="Hugging Face model id/path for --backend xgrammar-hf.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("HF_DEVICE", "auto"),
        help="Device for --backend xgrammar-hf: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--print-grammar",
        action="store_true",
        help="Print the XGrammar EBNF grammar and exit.",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print the raw model response to stderr before extracting code.",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Print generated code even if static validation reports errors.",
    )
    parser.set_defaults(strict=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.print_grammar:
        print(TRITON_MODULE_EBNF.strip())
        return 0

    try:
        if args.backend == "ollama":
            code = generate_with_ollama(args)
        else:
            code = generate_with_xgrammar_hf(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    errors = _static_validate_code(code, USER_PROMPT)
    print(code, end="" if code.endswith("\n") else "\n")
    if errors:
        print("\nStatic validation errors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2 if args.strict else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
