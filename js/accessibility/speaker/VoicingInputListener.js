// Copyright 2021, University of Colorado Boulder

/**
 * Trying out a single listener to be added to Display that will manage all input related to the voicing
 * feature and behave accordingly.
 *
 * Do not use yet, it is not finished.
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import scenery from '../../scenery.js';

class VoicingInputListener {

  /**
   * @param {Display} display
   */
  constructor( display ) {
    this.display = display;
  }

  // @public
  focus( event ) {
    const voicingNode = this.findVoicingNode( event.trail );

    if ( voicingNode && voicingNode.voicingFocusName ) {
      this.display.voicingUtteranceQueue.addToBack( voicingNode.voicingFocusName );
    }
  }

  // @private
  findVoicingNode( trail ) {
    let voicingNode = null;
    for ( let i = 0; i < trail.length; i++ ) {
      if ( trail.nodes[ i ].voicing ) {
        voicingNode = trail.nodes[ i ];
      }
    }

    return voicingNode;
  }
}

scenery.register( 'VoicingInputListener', VoicingInputListener );
export default VoicingInputListener;
