/**
 * CARL Cursor bridge — JSON-RPC server over stdin/stdout.
 *
 * The Python `CursorAdapter` spawns this process, sends `run_episode` requests
 * as newline-delimited JSON, and reads back episode results. This file is a
 * Day-1 stub: the protocol is defined and validated; calls into `@cursor/sdk`
 * land Day 4 of the build sequence.
 */

import * as readline from "node:readline";

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number | string;
  method: string;
  params: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number | string | null;
  result?: unknown;
  error?: { code: number; message: string };
}

const send = (resp: JsonRpcResponse): void => {
  process.stdout.write(JSON.stringify(resp) + "\n");
};

const rl = readline.createInterface({ input: process.stdin });

rl.on("line", (line: string) => {
  let req: JsonRpcRequest;
  try {
    req = JSON.parse(line) as JsonRpcRequest;
  } catch (err) {
    send({
      jsonrpc: "2.0",
      id: null,
      error: { code: -32700, message: "Parse error" },
    });
    return;
  }

  switch (req.method) {
    case "ping":
      send({ jsonrpc: "2.0", id: req.id, result: { pong: true } });
      break;

    case "run_episode":
      // Day-1: protocol-only stub. Day 4 wires @cursor/sdk.
      send({
        jsonrpc: "2.0",
        id: req.id,
        error: {
          code: -32601,
          message:
            "run_episode is a Day-4 stub. @cursor/sdk integration pending.",
        },
      });
      break;

    default:
      send({
        jsonrpc: "2.0",
        id: req.id,
        error: { code: -32601, message: `Unknown method: ${req.method}` },
      });
  }
});
