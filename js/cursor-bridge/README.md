# @carl-loop/cursor

Cursor adapter bridge for CARL. Spawned as a subprocess by the Python
`CursorAdapter`, this Node.js process speaks newline-delimited JSON-RPC 2.0
on stdin/stdout. Day 4 of the build sequence wires `@cursor/sdk` for actual
episode execution; the Day-1 stub validates the protocol.

```bash
cd js/cursor-bridge
npm install
npm run build
echo '{"jsonrpc":"2.0","id":1,"method":"ping"}' | node dist/index.js
```

The full protocol is documented in [`docs/multi_ide.md`](../../docs/multi_ide.md).
