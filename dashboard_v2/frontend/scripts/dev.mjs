import { spawn } from 'node:child_process'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const frontendRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const dashboardRoot = resolve(frontendRoot, '..')
const vitePath = resolve(frontendRoot, 'node_modules', 'vite', 'bin', 'vite.js')
const children = []
let stopping = false

function stop(exitCode = 0) {
  if (stopping) return
  stopping = true
  for (const child of children) {
    if (child.exitCode === null && child.signalCode === null) child.kill()
  }
  process.exitCode = exitCode
}

function startChild(name, command, args, cwd) {
  const child = spawn(command, args, { cwd, stdio: 'inherit' })
  children.push(child)
  child.on('error', (error) => {
    console.error(`[dev] Could not start ${name}: ${error.message}`)
    stop(1)
  })
  child.on('exit', (code, signal) => {
    if (stopping) return
    const result = signal ? `signal ${signal}` : `exit code ${code}`
    console.error(`[dev] ${name} stopped with ${result}`)
    stop(code ?? 1)
  })
  return child
}

async function waitForApi(api) {
  const deadline = Date.now() + 60_000
  while (Date.now() < deadline) {
    if (stopping || api.exitCode !== null || api.signalCode !== null) {
      throw new Error('API server exited before becoming ready.')
    }

    try {
      const response = await fetch('http://127.0.0.1:8000/v1/health', {
        signal: AbortSignal.timeout(1_000),
      })
      if (response.ok) return
    } catch {
      // The API may need a moment to initialize before its health route responds.
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 500))
  }
  throw new Error('API server did not become ready at http://127.0.0.1:8000/v1/health within 60 seconds.')
}

async function hasExistingVite() {
  let response
  try {
    response = await fetch('http://localhost:5173/', {
      signal: AbortSignal.timeout(2_000),
    })
  } catch (error) {
    if (error.cause?.code === 'ECONNREFUSED') return false
    throw new Error(`Could not check the dev server on port 5173: ${error.message}`)
  }

  const html = await response.text()
  if (response.ok && html.includes('/@vite/client')) return true
  throw new Error('Port 5173 is occupied by a service that is not this Vite dev server; refusing to use another port.')
}

function openDashboard() {
  if (process.platform !== 'win32' || !process.argv.slice(2).includes('--open')) return
  const browser = spawn('cmd.exe', ['/c', 'start', '', 'http://localhost:5173/'], {
    detached: true,
    stdio: 'ignore',
    windowsHide: true,
  })
  browser.unref()
}

async function main() {
  const api = startChild(
    'API server',
    process.env.PYTHON || 'python',
    ['api_server.py', '--host', '127.0.0.1', '--port', '8000', '--dashboard-root', '.'],
    dashboardRoot,
  )

  try {
    await waitForApi(api)
  } catch (error) {
    console.error(`[dev] ${error.message}`)
    stop(1)
    return
  }

  let reuseVite
  try {
    reuseVite = await hasExistingVite()
  } catch (error) {
    console.error(`[dev] ${error.message}`)
    stop(1)
    return
  }

  if (reuseVite) {
    console.log('[dev] Reusing the existing Vite dev server on http://localhost:5173/')
    openDashboard()
    return
  }

  startChild(
    'Vite',
    process.execPath,
    [vitePath, '--host', 'localhost', '--port', '5173', '--strictPort', ...process.argv.slice(2)],
    frontendRoot,
  )
}

process.on('SIGINT', () => stop())
process.on('SIGTERM', () => stop())
main().catch((error) => {
  console.error(`[dev] ${error.message}`)
  stop(1)
})
