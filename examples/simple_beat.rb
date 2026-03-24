# Simple beat example for the Sonic Pi → SC NRT transpiler
use_bpm 120

# Kick + snare pattern
live_loop :drums do
  sample :bd_haus, amp: 1.0
  sleep 0.5
  sample :sn_dub, amp: 0.7
  sleep 0.5
end

# Melody
live_loop :melody do
  with_fx :reverb, room: 0.6 do
    use_synth :saw
    play chord(:C4, :minor).choose, amp: 0.5, release: 0.3
    sleep 0.25
    play chord(:C4, :minor).choose, amp: 0.4, release: 0.2
    sleep 0.25
    play chord(:Eb4, :major).choose, amp: 0.5, release: 0.4
    sleep 0.5
  end
end
