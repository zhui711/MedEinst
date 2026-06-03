<script setup>
import { Collection, Document, Link, Tickets } from '@element-plus/icons-vue'
import { affiliationLine, authors, links } from '@/data/project'

const buttons = [
  { label: 'Paper', href: links.paper, icon: Document },
  { label: 'Code', href: links.code, icon: Link },
  { label: 'Dataset', href: links.dataset, icon: Collection },
  { label: 'Slides', href: links.slides, icon: Document },
  { label: 'BibTeX', href: '#citation', icon: Tickets, local: true },
]
</script>

<template>
  <header class="hero">
    <div class="hero-inner">
      <p class="venue-badge">Accepted to ACL 2026</p>
      <h1>
        MedEinst: Benchmarking the Einstellung Effect in Medical LLMs through
        Counterfactual Differential Diagnosis
      </h1>

      <nav class="author-list" aria-label="Authors">
        <a
          v-for="author in authors"
          :key="author.name"
          :href="author.homepage"
          target="_blank"
          rel="noreferrer"
        >
          {{ author.name }}
        </a>
      </nav>

      <p class="affiliation-line">{{ affiliationLine }}</p>

      <div class="hero-actions" aria-label="Project links">
        <a
          v-for="button in buttons"
          :key="button.label"
          class="action-button"
          :class="{ secondary: button.local }"
          :href="button.href"
          :target="button.local ? undefined : '_blank'"
          :rel="button.local ? undefined : 'noreferrer'"
        >
          <component :is="button.icon" class="button-icon" aria-hidden="true" />
          <span>{{ button.label }}</span>
        </a>
      </div>
    </div>
  </header>
</template>
