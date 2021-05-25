// Copyright 2020, University of Colorado Boulder

/**
 * A focus highlight for the voicing prototype. Has a unique color to indicate that focus is not around something
 * that is interactive, but can be read with activation.
 *
 * This should generally be used for otherwise NON interactive things that have voicing. Normally focusable things
 * should have the default focus highlight.
 * @author Jesse Greenberg
 */

import merge from '../../../../phet-core/js/merge.js';
import FocusHighlightFromNode from '../FocusHighlightFromNode.js';
import sceneryPhet from '../../../../scenery-phet/js/sceneryPhet.js';

class VoicingHighlight extends FocusHighlightFromNode {
  constructor( node, options ) {

    options = merge( {
      outerStroke: 'grey',
      innerStroke: 'black'
    }, options );

    super( node, options );
  }
}

sceneryPhet.register( 'VoicingHighlight', VoicingHighlight );
export default VoicingHighlight;