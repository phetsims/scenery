// Copyright 2026, University of Colorado Boulder

/**
 * Tracks global response groups used by ParallelDOM. Each group holds a reusable Utterance so
 * new responses on the same group replace earlier ones without interrupting other groups.
 *
 * This central registry avoids per-Node state and provides basic safeguards against unbounded growth.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Utterance from '../../../../utterance-queue/js/Utterance.js';
import scenery from '../../scenery.js';

// An arbitrary limit to prevent unbounded growth of the registry. The registry is meant for a stable set of
// groups. Dynamic creation indicates misuse or a memory leak.
const MAX_GROUPS = 50;

class ResponseGroupRegistry {

  // The map that stores the set of Utterances assigned to each group.
  private readonly groupMap = new Map<string, Utterance>();

  /**
   * Return a stable Utterance for a group. Creates one if it does not exist.
   *
   * @param group - A stable, global identifier for a self-interrupting response group.
   * @param defaultInterruptible - The default interruptible value for a new Utterance.
   */
  public getOrCreateGroupUtterance( group: string, defaultInterruptible: boolean ): Utterance {
    assert && assert( group.trim().length > 0, 'group must be a non-empty string' );

    let utterance = this.groupMap.get( group );
    if ( !utterance ) {
      utterance = new Utterance( {
        interruptible: defaultInterruptible,

        // A delay to allow grouped information to collect in the queue for fast interactions. But not too
        // long to cause a sluggish delay for the user. See https://github.com/phetsims/scenery/issues/1777
        alertStableDelay: 1000
      } );
      this.groupMap.set( group, utterance );

      assert && assert(
        this.groupMap.size <= MAX_GROUPS,
        `Too many response groups (${this.groupMap.size}). Groups should be stable, not dynamic.`
      );
    }
    return utterance;
  }

  /**
   * Remove a single group from the registry.
   */
  public removeGroup( group: string ): void {
    this.groupMap.delete( group );
  }

  /**
   * Clear all groups. Use only as a cleanup escape hatch.
   */
  public clearGroups(): void {
    this.groupMap.clear();
  }
}

const responseGroupRegistry = new ResponseGroupRegistry();

scenery.register( 'responseGroupRegistry', responseGroupRegistry );

export default responseGroupRegistry;
