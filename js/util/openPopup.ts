// Copyright 2022-2023, University of Colorado Boulder

/**
 * Opens a URL in a popup window or tab if possible.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import { allowLinksProperty } from '../imports.js';

/**
 * Opens the URL in a new window or tab.
 * @param url
 * @param allowPopups - Don't allow openPopup IF we have query parameters AND allowLinks is false,
 *                   - see https://github.com/phetsims/joist/issues/830
 *                   - But individual cases (such as screenshot) can override this to be always allowed
 */
function openPopup( url: string, allowPopups = allowLinksProperty.value ): void {

  // If available, don't openPopups for fuzzing
  const fuzzOptOut = phet && phet.chipper && phet.chipper.isFuzzEnabled();

  if ( allowPopups && !fuzzOptOut ) {
    const popupWindow = window.open( url, '_blank' ); // open in a new window/tab

    // We can't guarantee the presence of a window object, since if it isn't opened then it will return null.
    // See https://github.com/phetsims/phet-ios-app/issues/508#issuecomment-520891177 and documentation at
    // https://developer.mozilla.org/en-US/docs/Web/API/Window/open.
    popupWindow && popupWindow.focus();
  }
}

scenery.register( 'openPopup', openPopup );
export default openPopup;
