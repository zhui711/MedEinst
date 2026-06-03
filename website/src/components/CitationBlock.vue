<script setup>
import { ref } from 'vue'
import { Check, CopyDocument } from '@element-plus/icons-vue'
import { bibtex } from '@/data/project'

const copied = ref(false)

function fallbackCopy() {
  const textArea = document.createElement('textarea')
  textArea.value = bibtex
  textArea.setAttribute('readonly', '')
  textArea.style.position = 'fixed'
  textArea.style.top = '-1000px'
  document.body.appendChild(textArea)
  textArea.select()
  document.execCommand('copy')
  document.body.removeChild(textArea)
}

async function copyBibtex() {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(bibtex)
    } else {
      fallbackCopy()
    }
  } catch {
    fallbackCopy()
  }

  copied.value = true
  window.setTimeout(() => {
    copied.value = false
  }, 1800)
}
</script>

<template>
  <div class="citation-panel">
    <div class="citation-header">
      <p>Use the supplied BibTeX entry below.</p>
      <button type="button" class="copy-button" aria-live="polite" @click="copyBibtex">
        <component :is="copied ? Check : CopyDocument" class="button-icon" aria-hidden="true" />
        <span>{{ copied ? 'Copied' : 'Copy BibTeX' }}</span>
      </button>
    </div>
    <pre><code>{{ bibtex }}</code></pre>
  </div>
</template>
