// Copyright 2019-2022, University of Colorado Boulder

/**
 * Opens a URL in a popup window or tab if possible.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import phetCore from './phetCore.js';

/**
 * Opens the URL in a new window or tab.
 *
 * @param {string} url
 */
function openPopup( url ) {

  // Don't allow openPopup IF we have query parameters AND allowLinks is false,
  // see https://github.com/phetsims/joist/issues/830
  if ( !( window?.phet?.chipper?.queryParameters ) || ( window?.phet?.chipper?.queryParameters?.allowLinks ) ) {
    const popupWindow = window.open( url, '_blank' ); // open in a new window/tab

    // We can't guarantee the presence of a window object, since if it isn't opened then it will return null.
    // See https://github.com/phetsims/phet-ios-app/issues/508#issuecomment-520891177 and documentation at
    // https://developer.mozilla.org/en-US/docs/Web/API/Window/open.
    popupWindow && popupWindow.focus();
  }
}

phetCore.register( 'openPopup', openPopup );
export default openPopup;