import type { EngineInterface, Register } from 'claude-code'

// memlab's tools, from the user-scope MCP server or the memlab plugin's.
const MEMLAB = /^mcp__(plugin_memlab_)?memlab__/

const norm = (p: string) => p.trim().replace(/\\/g, '/').replace(/\/+$/, '').toLowerCase()

// The session's git worktree when it is not the main checkout, else null.
async function findWorktree($: EngineInterface): Promise<string | null> {
  // process.run runs in the session's directory by default
  const top = await $.process.run(['git', 'rev-parse', '--show-toplevel'])
  const common = await $.process.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'])
  if (top.exitCode !== 0 || common.exitCode !== 0) return null
  const main = common.stdout.trim().replace(/\\/g, '/').replace(/\/\.git\/?$/, '')
  return norm(top.stdout) === norm(main) ? null : top.stdout.trim()
}

export const register: Register = on => {
  // asked once per load
  let worktree: Promise<string | null> | undefined
  let calls = 0

  on('tool.call', async ($, e, next) => {
    if (!MEMLAB.test(e.tool)) return next(e)
    worktree ??= findWorktree($).catch(() => null)
    const wt = await worktree
    // Desktop-app worktree sessions get a server indexing the main checkout: send the worktree.
    const args = e as typeof e & { root?: string }
    const ran = await next(wt && !args.root ? { ...e, root: wt } : e)
    calls += 1
    const where = wt ? `worktree ${wt.replace(/\\/g, '/').split('/').pop()}` : 'main checkout'
    $.ui.status(`memlab · ${where} · ${calls} call${calls === 1 ? '' : 's'}${ran.isError ? ' · last call failed' : ''}`)
    return ran
  }).catch(($, e, next) => next(e))   // a failed lookup must never block the memlab call
}
