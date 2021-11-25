// Copyright 2018-2021, University of Colorado Boulder

/**
 * Utilities for full-screen support
 * Used to live at '/joist/js/FullScreen'. Moved to '/scenery/js/util/FullScreen' on 4/10/2018
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import detectPrefix from '../../../phet-core/js/detectPrefix.js';
import detectPrefixEvent from '../../../phet-core/js/detectPrefixEvent.js';
import platform from '../../../phet-core/js/platform.js';
import { scenery } from '../imports.js';

// get prefixed (and properly capitalized) property names
const exitFullscreenPropertyName = detectPrefix( document, 'exitFullscreen' ) ||
                                   detectPrefix( document, 'cancelFullScreen' ); // Firefox
const fullscreenElementPropertyName = detectPrefix( document, 'fullscreenElement' ) ||
                                      detectPrefix( document, 'fullScreenElement' ); // Firefox capitalization
const fullscreenEnabledPropertyName = detectPrefix( document, 'fullscreenEnabled' ) ||
                                      detectPrefix( document, 'fullScreenEnabled' ); // Firefox capitalization
let fullscreenChangeEvent = detectPrefixEvent( document, 'fullscreenchange' );

// required capitalization workaround for now
if ( fullscreenChangeEvent === 'msfullscreenchange' ) {
  fullscreenChangeEvent = 'MSFullscreenChange';
}

const FullScreen = {

  // @public
  isFullScreen() {
    return !!document[ fullscreenElementPropertyName ];
  },

  // @public
  isFullScreenEnabled() {
    return document[ fullscreenEnabledPropertyName ] && !platform.safari7;
  },

  /**
   * @public
   * @param {Display} display
   */
  enterFullScreen( display ) {
    const requestFullscreenPropertyName = detectPrefix( document.body, 'requestFullscreen' ) ||
                                          detectPrefix( document.body, 'requestFullScreen' ); // Firefox capitalization

    display.domElement[ requestFullscreenPropertyName ] && display.domElement[ requestFullscreenPropertyName ]();
  },

  // @public
  exitFullScreen() {
    document[ exitFullscreenPropertyName ] && document[ exitFullscreenPropertyName ]();
  },

  /**
   * @public
   * @param {Display} display
   */
  toggleFullScreen( display ) {
    if ( FullScreen.isFullScreen() ) {
      FullScreen.exitFullScreen();
    }
    else {
      FullScreen.enterFullScreen( display );
    }
  },

  isFullScreenProperty: new Property( false )
};

// update isFullScreenProperty on potential changes
document.addEventListener( fullscreenChangeEvent, evt => {
  FullScreen.isFullScreenProperty.set( FullScreen.isFullScreen() );
} );

scenery.register( 'FullScreen', FullScreen );
export default FullScreen;