# Python API

The stable user-facing workflow is available from `import hyperspatial as hs`:

```python
hs.design_panel(...)
hs.adapt(...)
hs.explore(...)
hs.forecast(...)
hs.validate(...)
hs.inspect_run(...)
```

The distribution is named `hyperspatial-map` and the import namespace remains `hyperspatial` for compatibility with the frozen analyses. The primary command-line executable is `atlastailor`; `hyperspatial` remains a legacy alias.

Use the API sections for detailed signatures. Functions not exported from `hyperspatial.__all__` should be treated as implementation details unless explicitly documented.

::: hyperspatial
