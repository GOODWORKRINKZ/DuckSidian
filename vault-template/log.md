# Log

Хронологический лог операций агента. Append-only. Каждая запись начинается с
`## [YYYY-MM-DD HH:MM] <op> | <details>`.

Полезные команды:
```
grep "^## \[" log.md | tail -20
grep "ingest" log.md | tail -10
```
