# shortcut-learning-vision
Investigating Shortcut Learning in Deep Vision Models

## Branch Convention

### 브랜치 구조
```
main
├── feat/data-loading        # 데이터 로딩
├── feat/erm                 # ERM Baseline
├── feat/group-dro           # Group DRO
├── feat/bg-randomization    # Background Randomization
└── feat/gradcam             # Grad-CAM 시각화
```

### 규칙
- `main` 브랜치에는 완성된 코드만 올립니다
- 각자 자신의 기능 브랜치에서 작업합니다
- 완성되면 main에 merge합니다

### 브랜치 생성 방법
```
git checkout -b feat/브랜치이름
```

### 작업 후 push 방법
```
git add .
git commit -m "add: 작업내용"
git push origin feat/브랜치이름
```

### 커밋 메시지 규칙
| 태그 | 설명 |
|------|------|
| add | 새 파일 추가 |
| fix | 버그 수정 |
| done | 기능 완성 |
| docs | 문서 수정 |

### 역할 분담
| 브랜치 | 담당자 |
|--------|--------|
| feat/data-loading | 찬민 |
| feat/erm | 찬민 |
| feat/group-dro | 찬민 |
| feat/bg-randomization | 태기 |
| feat/gradcam | 태기 |
