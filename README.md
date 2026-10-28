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
| Recover robot | 보호정지 해제 후 상태·정지 확인 → 정책 버퍼 초기화 → READY |
| Save `[W]` | 결과와 notes 저장 |
| Delete last `[D]` | 확인 후 마지막 저장 결과/영상 삭제 |
| Exit `[Esc]` | pending action 취소 → 미분류 attempt 폐기 → 장치 종료 |

`max_steps`에 도달하면 정지 후 operator outcome을 기다린다. 자동 성공/실패로 분류하지 않는다.
target Hz와 max steps는 READY에서 수정할 수 있다. 지연된 추론 결과는 Stop 이후 실행되지 않는다.
성공/실패 뒤 home의 완료·취소·오류도 기록한다. 통신/장치 오류는 수동 task 실패와 구분한다.

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
선택 패널은 기본으로 접혀 있으며 헤더의 `Task / checkpoint...`로 펼친다. Load 요청을 수락하면
다시 접어 결과표·notes 공간을 확보한다. 로드 오류는 패널을 열어 전체 traceback을 보여주며
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

잘못된 경로나 정책 로드 실패는 선택을 수정해 재시도할 수 있다. 추론·장치 오류, 정지 미확인,
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

adapter의 관측·home·episode 시작·명령 중 `RobotRecoveryRequired(reason, status=...)`가 발생하면
아직 수동 분류하지 않은 rollout은 오류로 보존하고 성공률에서는 제외한다. 이미 Success/Failure를
저장한 뒤 복귀 home에서 발생한 오류는 수동 결과를 유지하고 `outcome_home='error'`로 기록한다.
세션은 RECOVERY_REQUIRED로 전환하며
GUI·정책·카메라를 유지한다. Start, 성공/실패 분류, 자동 home과 checkpoint 교체는 차단한다.
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
results.json / notes.txt        수동 결과·성공률과 notes
rollout_0001_.../
  result.json                   outcome, home 상태, 실제 Hz, latency 통계
  steps.jsonl                   state/action/command/그리퍼와 단계별 시간
  <camera-role>.mp4            실제 capture 시간에 대응하는 영상
  frames.jsonl                  source 관측 step과 video frame 매핑
```

actual command Hz는 `(commands-1)/(last_command_time-first_command_time)`로 계산한다.
policy/client/RPC/server/GPU 지연은 제공된 실제 측정값만 구분해서 집계한다.
Capture, command, 영상 encoding 시간과 deadline miss도 따로 저장한다.

영상은 policy가 본 관측을 monotonic capture 시간으로 30fps에 재표본화한다. 새 관측 전까지
이전 이미지를 유지하고 마지막 pending 이미지는 종료 시 기록한다. target Hz로 영상 fps를
가정하지 않는다. 독립 observer camera의 연속 녹화는 별도 adapter가 필요한 확장이다.

시작 장면 browser는 호출 프로젝트의 `reference_manifest`를 사용한다. manifest는
`task/instruction/episode_count`와 step0 sample의 primary/wrist 파일 및 optional checksum을 포함한다.
다른 task, 경로 탈출과 checksum 불일치를 거절하며 reference 이미지는 policy 입력이 아니다.
