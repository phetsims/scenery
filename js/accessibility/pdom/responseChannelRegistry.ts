// Copyright 2026, University of Colorado Boulder

/**
 * Tracks global response channels used by ParallelDOM. Each channel holds a reusable Utterance so
 * new responses on the same channel replace earlier ones without interrupting other channels.
 *
 * This central registry avoids per-Node state and provides basic safeguards against unbounded growth.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Utterance from '../../../../utterance-queue/js/Utterance.js';
import scenery from '../../scenery.js';

// An arbitrary limit to prevent unbounded growth of the registry. The registry is meant for a stable set of
// channels. Dynamic creation indicates misuse or a memory leak.
const MAX_CHANNELS = 50;

class ResponseChannelRegistry {

  // The map that stores the set of Utterances assigned to each channel.
  private readonly channelMap = new Map<string, Utterance>();

  /**
   * Return a stable Utterance for a channel. Creates one if it does not exist.
   *
   * @param channel - A stable, global identifier for a self-interrupting response channel.
   * @param defaultInterruptible - The default interruptible value for a new Utterance.
   */
  public getOrCreateChannelUtterance( channel: string, defaultInterruptible: boolean ): Utterance {
    assert && assert( channel.trim().length > 0, 'channel must be a non-empty string' );

    let utterance = this.channelMap.get( channel );
    if ( !utterance ) {
      utterance = new Utterance( { interruptible: defaultInterruptible } );
      this.channelMap.set( channel, utterance );

      assert && assert(
        this.channelMap.size <= MAX_CHANNELS,
        `Too many response channels (${this.channelMap.size}). Channels should be stable, not dynamic.`
      );
    }
    return utterance;
  }

  /**
   * Remove a single channel from the registry.
   */
  public removeChannel( channel: string ): void {
    this.channelMap.delete( channel );
  }

  /**
   * Clear all channels. Use only as a cleanup escape hatch.
   */
  public clearChannels(): void {
    this.channelMap.clear();
  }
}

const responseChannelRegistry = new ResponseChannelRegistry();

scenery.register( 'responseChannelRegistry', responseChannelRegistry );

export default responseChannelRegistry;
