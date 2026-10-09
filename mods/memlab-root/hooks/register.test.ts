import { test, expect } from 'claude-code/testing'

const git = (cwd: string, main: string) => (argv: readonly string[]) => {
  const arg = argv[argv.length - 1]
  if (arg === '--show-toplevel') return { exitCode: 0, stdout: cwd + '\n', stderr: '', isStdoutTruncated: false, isStderrTruncated: false }
  return { exitCode: 0, stdout: main + '/.git\n', stderr: '', isStdoutTruncated: false, isStderrTruncated: false }
}

for (const [name, cwd, wantRoot] of [
  ['a worktree session adds its worktree as root', 'E:/repo/.claude/worktrees/wt1', 'E:/repo/.claude/worktrees/wt1'],
  ['the main checkout leaves the call as it was', 'E:/repo', undefined],
] as const) {
  test(name, async ($, on) => {
    const seen: Record<string, unknown>[] = []
    on('ui.status', () => ({ value: undefined }))
    on('process.run', ($, e) => ({ value: git(cwd, 'E:/repo')(e.argv) }))
    on('tool.call', ($, e) => { seen.push(e as Record<string, unknown>); return { result: 'ok' } })

    await $.tool.call({ tool: 'mcp__memlab__search_code', query: 'x' })
    await $.tool.call({ tool: 'mcp__plugin_memlab_memlab__search_decisions', query: 'y', root: 'E:/other' })
    await $.tool.call({ tool: 'Read', file_path: 'a.txt' })

    expect(seen[0].root).toBe(wantRoot)
    expect(seen[1].root).toBe('E:/other')          // an explicit root is kept
    expect(seen[2].root).toBe(undefined)           // other tools untouched
  })
}
