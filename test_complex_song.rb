# Kompleks Sonic Pi parçası — transpiler stres testi

use_bpm 120
use_synth :piano

# --- Teori yardımcıları ---
def chord_progression(root, degrees)
  degrees.map { |d| note(root) + d }
end

def humanize(val, amount=0.05)
  val + rrand(-amount, amount)
end

# --- Değişkenler ---
base_note = :C3
progression = [0, 5, 7, 3]   # C - F - G - Eb (semitone offsets)
bar = 0

# --- Melodi pattern ---
melody_notes = (scale :C4, :major).take(8)
melody_times = [0.25, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.75]

# --- Euclidean ritim ---
kick_pattern  = spread(3, 8)
snare_pattern = spread(2, 8, rotate: 2)
hihat_pattern = spread(5, 8)

# --- Arpejyatör ---
arp_chord = chord(:C4, :minor7)
arp_idx = 0

# --- Ana döngü (live_loop simülasyonu) ---
4.times do |bar_num|
  root_offset = progression[bar_num % progression.length]
  current_root = base_note.to_s

  # Bas
  with_synth :tb303 do
    with_fx :lpf, cutoff: 80 do
      play note(base_note) + root_offset, release: 0.3, amp: 0.8
    end
  end

  # Melodi
  melody_notes.each_with_index do |n, i|
    with_fx :reverb, room: 0.6 do
      play n + root_offset,
           amp: humanize(0.6),
           pan: rrand(-0.3, 0.3),
           release: melody_times[i] * 0.9
    end
    sleep melody_times[i]
  end

  # Akor vurgusu (her 2 barda bir)
  if bar_num.even?
    chord_notes = chord_progression(base_note, [0, 4, 7, root_offset])
    chord_notes.each do |cn|
      play cn, amp: 0.4, release: 1.5, attack: 0.1
    end
  end

  bar += 1
end

# --- Perküsyon bölümü ---
8.times do |i|
  kick_pattern.each do |hit|
    if hit
      sample :bd_haus, amp: 1.2, rate: rrand(0.98, 1.02)
    end
    sleep 0.25
  end
end

# --- Dinamik filtre sweep ---
cutoff_vals = (line 60, 120, steps: 16).to_a
cutoff_vals.each_with_index do |cv, i|
  with_fx :lpf, cutoff: cv do
    with_synth :saw do
      play_chord chord(:C3, :minor), amp: 0.5, release: 0.24
    end
  end
  sleep 0.25
end

# --- String interpolation ile log ---
song_length = melody_times.sum * 4
puts "Toplam melodi süresi: #{song_length} beat"
puts "Progression: #{progression.map { |p| p.to_s }.join(' - ')}"

# --- Hash ile konfig ---
synth_config = {
  amp: 0.7,
  attack: 0.05,
  release: 0.4,
  pan: 0.0
}

with_synth :blade do
  (scale :C4, :minor_pentatonic).each do |n|
    play n, **synth_config
    synth_config[:pan] = synth_config[:pan] + 0.1
    sleep 0.25
  end
end

# --- Lamda/proc kullanımı ---
note_transform = lambda { |n, offset| n + offset }
transposed = melody_notes.map { |n| note_transform.call(n, 12) }

transposed.zip(melody_times).each do |n, t|
  play n, amp: 0.5, release: t
  sleep t
end

# --- Case/when ile dinamik synth seçimi ---
[0, 1, 2, 3].each do |beat_type|
  synth_name = case beat_type
               when 0 then :beep
               when 1 then :piano
               when 2 then :saw
               else        :tri
               end
  with_synth synth_name do
    play :C4, amp: 0.6, release: 0.3
  end
  sleep 0.5
end

# --- Tick/ring ile döngüsel pattern ---
pattern_ring = ring(:C4, :E4, :G4, :B4, :A4, :F4)
rhythm_ring  = ring(0.25, 0.25, 0.5, 0.25, 0.5, 0.25)

12.times do
  n = pattern_ring.tick
  t = rhythm_ring.look
  play n, amp: 0.7, release: t * 0.8
  sleep t
end
