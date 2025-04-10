// Copyright 2025, University of Colorado Boulder

/**
 * A PressListener that speaks the name and hint responses of a VoicingNode when it is clicked. If there
 * is movement of the pressed pointer, the voicing response is interrupted.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Vector2 from '../../../../dot/js/Vector2.js';
import PressListener from '../../listeners/PressListener.js';
import scenery from '../../scenery.js';
import type { VoicingNode } from './Voicing.js';

// If the mouse moves this much in the global coordinate frame, we consider it a drag event and the voicing response
// behavior is interrupted.
const GLOBAL_DELTA = 1;

export default class VoicingActivationResponseListener extends PressListener {
  public constructor( voicingNode: VoicingNode ) {

    let startPosition: Vector2 | null = null;

    super( {

      // This listener should not attach to the Pointer and should not interfere with other listeners.
      attach: false,
      press: event => {
        startPosition = event.pointer.point;
      },
      drag: event => {

        if ( event && startPosition && startPosition.distance( event.pointer.point ) > GLOBAL_DELTA ) {
          this.interrupt();
        }
      },
      release: event => {

        // If there is a change in position, speak the name and hint of the voicing node.
        if ( event && !this.interrupted ) {
          voicingNode.voicingSpeakResponse( {
            nameResponse: voicingNode.voicingNameResponse,
            hintResponse: voicingNode.voicingHintResponse
          } );
        }
      }
    } );
  }
}

scenery.register( 'VoicingActivationResponseListener', VoicingActivationResponseListener );