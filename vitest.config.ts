import { playwright } from '@vitest/browser-playwright';
import { defineConfig } from 'vitest/config';

const executablePath = process.env.CHROMIUM_PATH ?? (process.env.CI ? undefined : '/opt/pw-browsers/chromium');

const browser = (include: string[], name: string, timeout: number) => ({
  extends: true as const,
  resolve: { conditions: ['source', 'browser', 'module', 'import', 'default'] },
  optimizeDeps: { exclude: ['@huggingface/transformers', 'onnxruntime-web', 'phonemizer'] },
  define: { __EDGEWISE_TEST_HUB__: JSON.stringify(process.env.EDGEWISE_TEST_HUB ?? '') },
  test: {
    name,
    include,
    fileParallelism: false,
    testTimeout: timeout,
    hookTimeout: timeout,
    browser: {
      enabled: true,
      headless: true,
      provider: playwright({
        launchOptions: {
          executablePath,
          args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-certificate-errors'],
        },
      }),
      instances: [{ browser: 'chromium' as const }],
    },
  },
});

export default defineConfig({
  test: {
    globals: true,
    projects: [
      {
        extends: true,
        resolve: { conditions: ['source', 'node', 'import', 'default'] },
        ssr: { resolve: { conditions: ['source', 'node', 'import', 'default'], externalConditions: ['source'] } },
        test: { name: 'node', environment: 'node', include: ['test/unit/**/*.test.ts'] },
      },
      browser(['test/unit/**/*.test.ts'], 'browser', 30_000),
      {
        extends: true,
        resolve: { conditions: ['source', 'node', 'import', 'default'] },
        ssr: { resolve: { conditions: ['source', 'node', 'import', 'default'], externalConditions: ['source'] } },
        test: {
          name: 'node-models',
          environment: 'node',
          include: ['test/models/**/*.test.ts'],
          testTimeout: 900_000,
          hookTimeout: 900_000,
          fileParallelism: false,
        },
      },
      browser(['test/models/**/*.test.ts'], 'browser-models', 900_000),
    ],
  },
});
