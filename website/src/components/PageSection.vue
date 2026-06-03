<script setup>
defineProps({
  id: {
    type: String,
    required: true,
  },
  title: {
    type: String,
    required: true,
  },
  lede: {
    type: String,
    default: '',
  },
  tone: {
    type: String,
    default: '',
  },
})
</script>

<template>
  <section :id="id" class="section-band" :class="{ 'section-band-soft': tone === 'soft' }">
    <div class="section-inner">
      <div class="section-heading">
        <h2>{{ title }}</h2>
        <p v-if="lede" class="section-lede">{{ lede }}</p>
      </div>
      <slot />
    </div>
  </section>
</template>
