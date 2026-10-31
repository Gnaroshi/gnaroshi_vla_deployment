# gnaroshi_vla_deployment

로봇·모델별 프로젝트에서 가져다 쓰는 공통 GUI 배포 skeleton이다. 공통 저장소에는 GUI,
rollout 상태 관리, 영상·측정·수동 결과 기록, adapter 계약과 CPU mock만 둔다.
재사용 가능한 UR5e hardware adapter도 포함하며 특정 모델·논문·태스크의 실행 코드는 없다.

```text
src/gnaroshi_deployment/
  contracts.py             Observation / Prediction / Policy / Embodiment 계약
  session.py               worker, rollout 상태, 취소, 수동 outcome
  gui.py                   관측 화면, 버튼, 설정, notes와 결과
  records.py               JSON/JSONL, 영상, latency/실제 command Hz
  reference.py             호출 프로젝트가 제공하는 시작 장면 탐색
  desktop.py               현재 사용자의 desktop 환경 탐색
  adapters/mock.py         장치·모델 없이 실행 가능한 예제
  adapters/ur5e.py          모델과 독립적인 hardware adapter
scripts/deploy_mock.sh      CPU mock GUI launcher
tests/test_session.py       상태·취소·영상·hardware 계약 회귀 검증
```

## Mock 실행

Python 3.10 이상, NumPy, Pillow, Tk가 필요하다. 영상에는 OpenCV가 필요하다.
기존 호환 환경을 사용하거나 `pip install -e '.[video]'`로 설치한다. Tk는 Python/OS 패키지다.

```bash
DEPLOYMENT_PYTHON=python3 \
DEPLOYMENT_OUTPUT_ROOT=/absolute/run-storage/mock \
  bash scripts/deploy_mock.sh
```

기본은 현재 DISPLAY의 mock GUI이며 모델, 정책 서버, 로봇, 카메라에 연결하지 않는다.
`DEPLOYMENT_OUTPUT_ROOT`를 생략하면 이 checkout의 ignored `.runtime/mock`에 기록한다.
Python CLI도 같은 환경변수 또는 `--output-root`로 경로를 지정하며 기본은 `.runtime/results`다.
SSH에서 GUI가 필요하면 대상 desktop 또는 trusted X11 forwarding을 사용한다.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m gnaroshi_deployment.cli \
  --output-root /absolute/run-storage/mock --headless-demo
```

`--headless-demo`는 mock rollout을 자동 실행해 outcome 저장·home 복귀를 검증한다.
실제 모델 factory와 함께 사용할 수 없다. 가상 display에서는 `--gui-smoke`와
`--screenshot /absolute/run-storage/gui.png`로 GUI 경로를 확인할 수 있다.

## 프로젝트에서 확장

공통 저장소는 skeleton으로 유지한다. 각 연구 프로젝트 안에 clone한 작업 사본에서
`<backbone>/<paper>/` 아래에 전용 코드를 관리한다. 실행 경로에 모델·논문 이름을 드러낸다.

```text
<project>/gnaroshi_vla_deployment/
  src/gnaroshi_deployment/       공통 skeleton
  <backbone>/<paper>/
    deploy.sh                   task/model/checkpoint/control 설정
    launch.py                   해당 정책 프로세스와 GUI의 lifecycle
    policy.py                   모델 전처리·추론·reset·latency adapter
    tasks.json                  해당 프로젝트의 instruction/home/장치 설정
```

1. `contracts.py`와 `adapters/mock.py`를 읽고 `(SessionConfig, Policy, Embodiment)`를 반환하는
   factory를 구현한다. `python -m gnaroshi_deployment.cli --factory MODULE:FUNCTION --config FILE`로 연결한다.
2. image role/RGB·resize, state/action 차원·좌표·단위·정규화, horizon과 실제 실행 길이,
   reset 및 checkpoint binding을 해당 모델의 실제 구현에서 확인한다. 다른 모델에서 추정하지 않는다.
3. 새 로봇은 `connect/observe/home/begin_episode/apply/stop/close` 계약을 구현한다.
   hardware adapter에는 모델 코드를 import하지 않는다. 카메라 역할은 `SessionConfig.camera_roles`로 지정한다.
4. launcher 상단에서 task, method, checkpoint, target Hz, max steps, camera와 output을 수정하게 한다.
   모델 이름·논문 이름·실행 mode가 GUI와 저장된 config에도 나타나야 한다.
5. 모델·runtime·checkpoint·cache는 소유 프로젝트에서 관리한다. 장치 주소, camera serial,
   task home과 로봇별 규약도 프로젝트가 공급한다. UR5e adapter는 이 값이 없으면 생성을 거절한다.
6. 현재 할당받은 GPU를 각 실행에서 명시한다. 컨테이너에 보이는 장치와 실제 사용 배정은 구분한다.
7. import/inspect → CPU tests → mock GUI → 장치 미연결 정책 preflight → operator 실험 순서로 검증한다.

프로젝트별 변경을 공통 skeleton의 기본 branch에 섞지 않는다. 공통 기능 개선만 분리해서 반영한다.

## GUI 동작

| 동작 | 결과 |
|---|---|
| Start `[N]` | task home → policy reset → fresh rollout |
| Success `[S]`, Failure `[F]` | 명령 중단 → 수동 결과/영상 저장 → task home |
| Stop `[X]`, Retry `[R]` | 현재 attempt 폐기 → 현 위치 유지 |
| Home `[H]` | READY에서 팔만 task home으로 이동; rollout·policy reset 없음 |
| Recover robot | 보호정지 해제 후 상태·정지 확인 → 정책 버퍼 초기화 → READY |
| Save `[W]` | 결과와 notes 저장; notes 편집도 자동 저장 |
| Devices | 카메라 검색·역할 변경·swap·카메라만 재연결 |
| Config | READY에서 실행 속도·명령 수·녹화 변경, 다음 Load에 적용할 정책 설정 편집 |
| Settings | 글자 크기·참고 장면 표시·pane 배치·rollout 연속 번호 설정 |
| Help `[F1]` | 단축키·운용 안내 |
| Delete last `[D]` | 확인 후 마지막 저장 결과/영상 삭제 |
| Exit `[Esc]` | pending action 취소 → 미분류 attempt 폐기 → 장치 종료 |

`max_steps`에 도달하면 정지 후 operator outcome을 기다린다. 자동 성공/실패로 분류하지 않는다.
target Hz와 max steps는 READY에서 수정할 수 있다. 지연된 추론 결과는 Stop 이후 실행되지 않는다.
성공/실패 뒤 home의 완료·취소·오류도 기록한다. 실행 중 보호정지·연결 끊김·IK 실패·정책 오류는
**자동 Failure**로 저장하고 성공률에 반영한다. 수동 Failure와는 원인·`outcome_source`로 구분한다.
대기 중 발생한 오류는 note와 event를 남기며 새 rollout을 만들지 않는다.

UR5e의 Start와 outcome 복귀는 그리퍼 열림을 실측한 뒤 팔을 이동하고, 모든 관절이 home
허용 오차 안에 연속으로 들어왔는지 확인한다. 명령 수락만으로 복귀 완료를 기록하지 않는다.
열림·관절 도달 확인에는 제한 시간이 있으며, 실패하면 원인을 기록하고 다음 Start를 막는다.
이미 저장한 Success/Failure는 복귀 오류로 바꾸지 않는다. Stop/Retry는 계속 현 위치에서
멈추고 attempt를 폐기하며, 자동으로 home을 실행하지 않는다.

Rollout 번호는 기본적으로 **저장한 결과에 연속으로 부여**한다. Stop/Retry로 폐기한 시도는
번호를 소비하지 않는다. Settings에서 연속 번호를 끄면 폐기한 시도까지 세는 attempt 번호를
사용한다. 설정은 다음 실행에도 유지되며, 진행 중에는 바꿀 수 없다. 기존 저장 결과의 번호와
폴더는 변경하지 않는다. 각 결과의 `attempt_id`와 event 기록으로 실제 시도 순서도 추적한다.

Home은 선택·로딩·실행·정지·복구 중에는 비활성화된다. UR5e는 팔의 자세만 이동하고 그리퍼를
열거나 activation하지 않는다. Stop/Exit로 취소할 수 있다. 다른 embodiment는 optional
`home_idle(cancel)`을 제공하면 같은 분리가 적용되며, 없으면 기존 `home(cancel)`을 사용한다.
단축키는 입력창·설정 선택·notes 편집 중에는 실행되지 않는다.

### 화면과 실패 원인

화면은 [gnaroshi_mds UI/UX 지침](https://github.com/Gnaroshi/gnaroshi_mds/blob/main/guides/ui-ux.md)의
작업용 dark palette, native control, semantic typography와 크기 조절 workspace를 따른다. 상태와 다음 조작, 카메라, 실행 결과를 먼저
보여주고 경로·원시 오류·checkpoint identity는 Details에서 확인한다. 색상과 함께 상태 문구를
표시한다. 자동 실패 원인은 rollout 목록과 읽기 전용 자동 notes에 남고 사용자가 작성한 notes와
별도로 저장한다. 사용자가 편집한 notes를 자동 메시지로 덮어쓰지 않는다.

### 집계·글자·작업 영역

로봇을 반복 평가하는 operator가 현재 상태와 다음 조작, 저장된 결과를 바로 읽도록 구성한다.
상단의 **Success / Failure / Success rate / Saved**는 현재 session에서 저장·확정한 rollout만
집계한다. 자동 fault도 Failure 합계에 포함하고, Failure 바로 아래에 그중 자동 실패 수를 표시한다.
Stop/Retry로 폐기한 attempt와 idle 오류는 제외한다.
Success 이후 복귀 Home 오류는 이미 확정된 outcome을 바꾸지 않는다. 0개일 때 성공률은
분모가 없으므로 `—`다. 다른 checkpoint를 Load하면 새 session의 집계를 보여주며 이전 결과는 보존한다.

집계와 실행 조작은 스크롤 영역 밖에 유지한다. 실행/정지, 판정/재시도, 로봇 복구를
자연스러운 버튼 너비의 세 그룹으로 배치하며 창이 좁으면 그룹 단위로 줄을 바꾼다.
집계는 같은 toolbar에 표시한다. 카메라·참고 장면·결과/설정·notes의 경계는
직접 끌어서 조절한다. 구분선에 Tab으로 focus를 주고 Arrow / Shift+Arrow로 크기를 바꾸거나
Home/End로 범위 끝, Enter 또는 더블클릭으로 기본 비율을 복원한다. 지원하는 pane은 숨김·펼침이
가능하며 다시 펼치는 조작은 접근 가능한 위치에 남는다. 좁은 창에서의 clamp는 넓은 창에 저장한
선호 비율을 지우지 않는다.

상단 A- / A+와 Settings의 글자 설정으로 읽기 크기를 조절한다. Ctrl+plus/minus로 확대·축소하고
Ctrl+0으로 기본값을 복원한다. 제목·본문·보조문구·수치는 named semantic font를 사용하며,
행 높이·control padding·줄바꿈·열 배치도 함께 바뀐다. EN/KO가 필요하면 OS에 해당 glyph를 가진
글꼴이 있어야 한다. semantic token의 원본은 gui.py의 THEME/SPACE와 font 역할이고, pane 동작과
local presentation persistence는 workspace.py에서 관리한다.
큰 글자로 낮은 창을 사용할 때는 긴 instruction과 policy 설명을 **Context**에서 펼쳐 볼 수 있다.
Policy 설정은 오른쪽 inspector에서 편집하므로 폼이나 로드 오류가 카메라를 아래로 밀지 않는다.
측정값은 내용에 필요한 높이만 사용한다. 학습 예시 manifest가 있으면 참고 장면을 기본으로
표시한다. Settings의 **Training examples**에서 명시적으로 바꾼 표시 여부는 다음 실행에도 유지한다.
상단 **Inspector**는 오른쪽 작업 영역을 숨기거나 펼친다.
**Save notes [W]**는 Notes에, 확인 후 삭제하는 **Delete last [D]**는 Rollouts에 둔다.
설정 입력칸으로 Tab 이동하면 중첩된 스크롤 영역도 해당 입력칸을 보이도록 따라간다.
휠은 포인터가 있는 입력 폼·텍스트·결과 목록을 스크롤한다. 닫힌 task/camera picker 위의 휠은
선택값을 바꾸지 않는다. Details 갱신은 읽던 위치·커서·선택을 유지하며, 선택한 텍스트는
선택을 해제할 때까지 보존한다.

최상단 상태 바는 준비·로딩·실행·정지·복구 단계와 다음 행동을 보여준다. 실행 중 진행 막대는
명령 수와 설정된 한도를 나타낸다. 한도 도달 뒤에는 수동 outcome을 기다리며 성공으로 판정하지 않는다.
로딩·Home·정지·복구처럼 전체 작업량이 정해지지 않은 단계에는 실제 단계명과 대기 표시를 사용한다.

Rollouts는 목록과 선택 결과/annotation을 나눈 크기 조절 패널이다. 가운데 경계를 끌거나
키보드로 조절할 수 있고 배치를 저장한다. 새 결과가 추가되면 마지막 행을 보여주며, 일반
상태 갱신은 읽던 위치와 선택을 바꾸지 않는다. 선택한 결과의 폴더·경로 복사·영상 열기는
같은 도구 영역에 두고, 마지막 결과 삭제는 대상 확인을 거치는 별도 파괴적 조작으로 구분한다.
Annotation은 여러 줄로 편집하며 자동 저장한다. Exit도 실행·판정 버튼과 구분해 표시한다.

글자·창·pane 배치는 local UI preferences에 보관한다. 글자 크기는 선택 즉시 저장하므로
checkpoint 교체나 다음 실행에서도 마지막 값을 복원한다. 저장하지 못하면 Settings에 오류를
표시하며 기존 파일은 보존한다. 실험 결과와 checkpoint 설정을 바꾸지 않는다.
caller는 DeploymentGUI의 optional preferences_path 또는 config.metadata의 ui_preferences_path로
경로를 지정할 수 있다. config_directory가 있으면 그 부모의 ui_preferences.json을 사용하고,
없으면 output_root 아래 ignored .ui_preferences.json을 사용한다. 설정 파일은 Git에 넣지 않는다.

프로젝트는 config.metadata의 display_method/display_task로 읽기 쉬운 이름을 제공할 수 있다.
로드한 policy.metadata의 display_method가 있으면 실제 모델 이름으로 우선한다. Canonical method와
checkpoint identity는 원본 그대로 기록하며 경로·hash·원시 오류는 Details에서 확인한다.
드물게 수정하는 설정은 reload field에 advanced=True를 지정해 Advanced settings에 넣는다.

### 같은 창에서 checkpoint 교체

프로젝트 launcher는 선택 항목과 새 세션 factory를 GUI에 전달할 수 있다.

```python
DeploymentGUI(root, session,
    reload_fields=[
        {"key": "checkpoint", "label": "Checkpoint", "kind": "directory"},
        {"key": "mode", "label": "Mode", "kind": "choice", "choices": ["default", "alternative"],
         "value": "default"},
    ],
    reload_factory=make_session)
# make_session(selection: dict, previous_config: SessionConfig) -> unstarted DeploymentSession
```

`kind`는 `directory`, `file`, `text`, `choice`를 지원한다. 초기 선택은
`config.metadata['reload_selection']`이 우선하며 각 field의 `value`가 기본값이다.
기본 `section='policy'`는 Policy에 표시하고, optional `section='config'`는 Config에 표시한다.
Config의 프로젝트별 field는 다음 **Load checkpoint**에 적용할 draft다. READY에서 바로 바꿀 수 있는
target Hz·max commands·녹화 설정은 같은 탭의 별도 Apply 영역에 둔다. Settings는 GUI 설정만 다룬다.
헤더의 **Policy**로 inspector의 Policy 탭을 연다. Load 요청을 수락하면 Rollouts 탭으로
돌아간다. Load 버튼은 긴 설정 폼의 스크롤 밖에 둔다. 로드 오류는 Policy 탭에 요약을,
Details에 전체 진단 정보를 보여주며
선택값과 오류를 해당 세션의 `events.jsonl`에도 기록한다.
factory는 선택한 checkpoint/method를 config와 policy metadata에 명시하고, 필요한 경우
checkpoint별 `output_root`를 정한다. factory에서 세션을 `start()`하지 않는다.
factory가 정규화한 `reload_selection`은 그대로 보존하고 입력한 값은 `reload_request`에 기록한다.
재로딩 후 선택 항목도 정규화한 경로와 설정으로 갱신한다.
factory가 실패하면 자체 생성한 서버·프로세스도 정리한 뒤 예외를 반환해야 한다.

Load checkpoint는 SELECTING, READY, 정상 종료된 세션 또는 정책 로드 단계에서 안전하게 종료된 ERROR에서
활성화된다. 실행·home·stop 중에는 교체할 수 없다. GUI는 notes를 저장하고 기존 세션의
worker 종료·장치 정지 확인·cleanup을 기다린 뒤 background thread에서 factory를 호출한다.
새 세션은 같은 camera roles를 유지하며 별도 결과 폴더를 사용한다. task/instruction/home은
프로젝트 factory가 검증한 새 config로 바꿀 수 있다. 창과 선택 항목은
유지하고 결과표·notes·측정·폴더 표시는 새 세션으로 바꾼다. 이전 결과·영상은 보존한다.

잘못된 경로나 정책 로드 실패는 선택을 수정해 재시도할 수 있다. 정책 연결·추론 오류는 Failure 저장과 정지 확인·소유 자원 정리가 모두 끝난 뒤 같은 창에서 명시적으로
다시 Load할 수 있다. 재로딩만으로 Home이나 rollout을 시작하지 않는다. 장치 명령 오류, 정지 미확인,
cleanup 실패는 교체를 차단한다. 로딩 중 Exit는 factory가 반환한 세션을 연결하지 않고 정리한 뒤
창을 닫는다. 상태 검사는 `session.can_replace`, 교체 직전 종료 확인은
`session.replacement_closed`, 종료 대기는 `session.shutdown_complete`로 구분한다.

프로필·태스크와 JSON 양식은 프로젝트가 다음 optional callbacks로 정의한다. 공통 GUI에는
논문별 기본값이나 설정 schema를 넣지 않는다.

```python
DeploymentGUI(root, session, reload_fields=fields, reload_factory=make_session,
    selection_resolver=resolve_selection,  # (selection, changed_key) -> dict
    config_loader=load_config,            # (Path) -> dict
    config_saver=save_config,             # (Path, selection) -> None
    config_directory=config_directory)
```

choice 변경과 Browse 뒤에는 field key로 resolver를 호출한다. JSON Load, 정책 Load, JSON Save는
각각 `config`, `load`, `save` key로 검증한다. 화면 field와 함께 `reload_selection`의 모든 숨은
설정도 전달한다. Runtime Apply는 context의 target Hz/max steps도 갱신한다. JSON Load는
선택 설정만 바꾸며 실제 적용은 Load checkpoint로 수행한다.

### 카메라·저장·결과 관리

Devices의 **Refresh connected cameras**는 버튼을 눌렀을 때만 연결 장치 이름·serial·지원 RGB
profile을 조회한다. 카메라 역할별로 serial을 선택하고 Apply cameras를 누르거나 Swap roles로
선택을 바꾼다. Swap은 draft만 바꾸며 Apply 전에는 실제 스트림을 바꾸지 않는다.
중복·미연결·비호환 카메라는 거절한다. 장치 검색은 모델·로봇·그리퍼를 연결하지 않는다.

카메라 변경·Restart camera preview는 idle에서 worker가 실행한다. 기존 로봇 정지를 확인하고
카메라 스트림만 바꾸며 **로드한 모델·이전 결과·notes를 유지**한다. 정책 buffer는 비운다.
실행·Home·복구 동작 중에는 변경을 막는다. 카메라 오류는 CAMERA_ERROR로 남고, 같은 창에서
카메라 재연결 후 별도의 Start로 재개한다. 진행 중 발생한 오류는 그때까지의 기록을 Failure로
보존한다. 카메라 복구가 로봇의 보호정지를 해제하거나 다음 rollout을 자동 시작하지 않는다.

Adapter는 optional `list_camera_devices()`와 `configure_cameras(serials, cancel)`를 구현한다.
전자는 `devices/selected/configuration/warnings`, 후자는 확인된 `selected/configuration/camera_metadata`
객체를 반환한다. enumeration과 stream open은 deadline이 있고, `camera_operation_pending`이 참이면
새 장치 작업·motion·자원 교체를 막는다. UR5e adapter는 frame 번호 반복, 오래된 host-comparable
sensor timestamp, 잘못된 RGB shape/type을 camera fault로 기록한다. Hardware clock을 host epoch와
직접 비교하지 않는다. 역할 identity와 실제 교체 내역은 manifest·event·rollout에 기록한다.

Config에서 영상 녹화 ON/OFF와 runtime 설정을 바꿀 수 있다. 출력 저장 공간과 녹화 상태,
preview 수신률·프레임 나이도 표시한다. 기본 free storage256MiB 미만이면 Home/첫 command 전에
Start를 막고, 실행 중 부족해지면 기록을 보존하며 정지한다. Warning 기본값은2GiB다.
`SessionConfig.metadata`의 `min_free_disk_bytes` / `warn_free_disk_bytes`로 프로젝트가 조정한다.
카메라 FPS, target Hz, 실제 command Hz, preview 수신률은 서로 다른 값이다.

Rollouts에서 결과를 선택해 폴더·카메라별 영상 열기, 경로 복사, rollout 메모를 사용할 수 있다.
메모는 자동 저장하며 outcome·원본 step log·영상을 바꾸지 않는다. 사용자 notes도 편집 시 자동
저장한다. 창의 checkpoint form과 설정 tab은 작은 화면에서 스크롤할 수 있다. Live image를
더블클릭하면 해당 관측 snapshot을 확대하고 F1으로 단축키를 확인한다.
Config의 **Saved profiles → Save config / Load config**는 프로젝트가 제공하는 JSON 저장·복원 기능이다.
JSON을 불러오는 동작은 draft만 바꾸며 정책을 자동으로 로드하지 않는다.

Ubuntu에서 기존 MP4의 디코더가 없으면 Open video가 선택 영상 하나의 VP8/WebM 호환본을
ignored `.runtime/playback`에 준비하고 desktop player로 연다. 원본 녹화·FPS·프레임 순서·제어 루프는
유지한다. 준비는 GUI 밖에서 수행하고 결과 목록 옆에 진행·오류를 표시한다. 재요청은 캐시를 사용하며,
종료 시 진행 중인 변환을 취소하고 소유 프로세스를 회수한다. 기존 녹화를 일괄 변환하지 않는다.

GPU/RPC 지연은 관측된 호출당 평균과 전체 실행 action당 적산 평균을 구분해 저장한다.
`amortized_ms_per_executed_action`을 이용하면 client-buffered step을 포함한 분모로 비교할 수 있다.
Backbone calls는 prefix 갱신과 expert 경로를 합한 횟수이며 전체 RPC 수와 구분한다.

### 정책 로딩 중 카메라 미리보기

`config.metadata['preview_before_policy']=True`이고 adapter가 `connect_preview/observe_preview`를
제공하면 카메라만 먼저 연다. 정책 로드는 별도 loader thread에서 수행하고 세션 worker는
카메라를 갱신한다. optional `policy.loading_status` 문자열 또는 callable의 실제 단계와 경과
시간을 GUI·journal에 표시한다. 정책 connect와 metadata 검증이 성공한 뒤에만 일반
`embodiment.connect()`로 로봇 제어를 연결한다. UR5e 미리보기는 RTDE·gripper를 연결하지 않는다.

미리보기 관측은 빈 state와 `preview_only=True` 표식을 가지며 policy 입력으로 거절한다.
Start는 READY에서만 활성화된다. Exit·카메라 오류·정책 실패 때에도 loader가 끝난 뒤 policy와
센서를 정리한다. 정책 adapter의 connect는 유한 deadline과 cancellation을 지원해야 한다.

처음 태스크부터 선택하려면 `config.metadata['defer_policy_load']=True`를 함께 지정한다.
GUI는 선택 패널을 펼치고 SELECTING에서 카메라만 갱신한다. 정책·로봇 제어는 연결하지 않는다.
Load factory는 선택한 config의 `defer_policy_load=False`로 새 세션을 반환한다. 이 opt-in이
없으면 기존처럼 자동 로드한다. 미리보기를 지원하지 않는 adapter는 SELECTING에서 영상 없이 대기한다.

UR5e 연결 단계는 gripper 상태를 읽으며 activation은 operator Start 이후에 수행한다.
activation 자체가 손가락 움직임을 만들 수 있다. 별도 Python open–close calibration 루프는 없다.
하드웨어 adapter의 실제 수행 능력은 CPU mock 검증과 별개다.

### 보호정지 뒤 GUI에서 복구

adapter의 관측·home·episode 시작·명령 중 `RobotRecoveryRequired(reason, status=..., category=...)`가 발생하면
진행 중인 미분류 rollout은 자동 Failure로 보존하고 성공률에 반영한다. 이미 Success/Failure를
저장한 뒤 복귀 home에서 발생한 오류는 수동 결과를 유지하고 `outcome_home='error'`로 기록한다.
세션은 RECOVERY_REQUIRED로 전환하며
GUI·정책·카메라를 유지한다. Start, 성공/실패 분류, Home과 checkpoint 교체는 차단한다.
지원되는 adapter에서는 camera-only preview를 계속 표시한다.

사용자가 티치펜던트에서 원인을 확인하고 보호정지를 해제한 뒤 Recover robot을 누른다.
worker는 이전 stop thread가 모두 끝난 것을 확인한 후 `embodiment.recover(cancel_event)`를
실행한다. 이 메서드는 새 상태와 정지 응답을 확인해 `{"healthy": True, "stop_confirmed": True}`를
반환해야 한다. 자동 unlock·전원·브레이크 해제·dashboard play·home·gripper activation·arming을
수행해서는 안 된다. 확인 실패는 원인을 표시하며 복구 대기 상태를 유지한다.

검증 성공 뒤 정책의 새 episode reset으로 이전 action buffer를 버리고 READY로 돌아간다.
모델을 다시 로드하거나 로봇을 이동하지 않는다. 다음 Start는 사용자가 별도로 누른다.
Stop·Exit는 복구를 취소하며 늦은 응답으로 재무장하지 않는다. 정책 오류와 일반 정지 미확인
오류는 이 경로를 활성화하지 않는다. 기존 sticky 실패 표시는 typed fault의 명시적 복구에서
새로운 정상 상태·정지 확인·정책 reset이 모두 성공했을 때에만 해제한다.

## 기록과 영상

```text
manifest.json                   config, source 상태, policy/robot/camera metadata
events.jsonl                    operator 요청, 폐기/삭제, 오류
results.json / notes.txt        수동·자동 결과, 성공률, 사용자·자동 notes
rollout_0001_.../
  result.json                   outcome/source, 실패 원인, home 상태, 실제 Hz, latency 통계
  steps.jsonl                   state/action/command/그리퍼와 단계별 시간
  failed_step.json               실패한 입력·시도 action·원인과 command 확인 여부
  <camera-role>.mp4            실제 capture 시간에 대응하는 영상
  frames.jsonl                  source 관측 step과 video frame 매핑
```

`failure_category`, `failure_reason`, `failure_note`, `failure_status`, `failure_phase`에
보호정지·로봇 연결 끊김·IK 해 없음·servo 거부 등의 실제 관측 원인을 남긴다. IK 실패에는
가능한 경우 action 크기와 target pose도 보관하며, 큰 action이 원인이라고 추정해서 쓰지 않는다.
실패 관측도 영상에 보관하되 확인되지 않은 명령을 실행된 step이나 latency 통계에 넣지 않는다.
기존 결과·영상은 소급 변경하지 않으며 복구를 다시 눌러도 같은 실패를 중복 집계하지 않는다.

actual command Hz는 `(commands-1)/(last_command_time-first_command_time)`로 계산한다.
policy/client/RPC/server/GPU 지연은 제공된 실제 측정값만 구분해서 집계한다.
Capture, command, 영상 encoding 시간과 deadline miss도 따로 저장한다.

영상은 policy가 본 관측을 monotonic capture 시간으로 30fps에 재표본화한다. 새 관측 전까지
이전 이미지를 유지하고 마지막 pending 이미지는 종료 시 기록한다. target Hz로 영상 fps를
가정하지 않는다. 독립 observer camera의 연속 녹화는 별도 adapter가 필요한 확장이다.

시작 장면 browser는 호출 프로젝트의 `reference_manifest`를 사용한다. manifest는
`task/instruction/episode_count`와 step0 sample의 primary/wrist 파일 및 optional checksum을 포함한다.
다른 task, 경로 탈출과 checksum 불일치를 거절하며 reference 이미지는 policy 입력이 아니다.
