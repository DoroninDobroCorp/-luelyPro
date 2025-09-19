// AudioWorklet: 2 inputs (near, room) -> 1 output (room only when near dominates)
// If near >> room -> assume speaker (ME) -> mute output so model hears only the room.

class GateProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.thresholdDb = options.processorOptions?.thresholdDb ?? 6; // near over room by ~6 dB
    this.hangMs = options.processorOptions?.hangMs ?? 250; // decision hold to avoid chattering
    this.sampleRate = sampleRate;
    this.hangSamples = Math.floor(this.hangMs * this.sampleRate / 1000);
    this.state = 'AUD'; // AUD | ME
    this.stateHold = 0;
  }

  // RMS -> dB
  _db(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) { const v = arr[i]; sum += v*v; }
    const rms = Math.sqrt(sum / Math.max(1, arr.length));
    return 20 * Math.log10(rms + 1e-9);
  }

  process(inputs, outputs) {
    const near = inputs[0]?.[0];
    const room = inputs[1]?.[0];
    const out = outputs[0][0];
    if (!near || !room) {
      out.set(room || near || new Float32Array(out.length));
      return true;
    }

    const dbNear = this._db(near);
    const dbRoom = this._db(room);

    const meLouder = (dbNear - dbRoom) > this.thresholdDb;
    if (meLouder) {
      this.state = 'ME';
      this.stateHold = this.hangSamples;
    } else if (this.stateHold > 0) {
      this.stateHold -= near.length;
    } else {
      this.state = 'AUD';
    }

    if (this.state === 'ME') {
      out.fill(0);
    } else {
      out.set(room);
    }

    this.port.postMessage({ t: currentTime, state: this.state, dbNear, dbRoom });
    return true;
  }
}

registerProcessor('gate-processor', GateProcessor);
