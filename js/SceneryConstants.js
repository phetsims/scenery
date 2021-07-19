// Copyright 2021, University of Colorado Boulder

/**
 * SceneryConstants is a collection of constants used throughout scenery.
 *
 * @author Chris Malley (PixelZoom, Inc.)
 */

import scenery from './scenery.js';

const SceneryConstants = {

  // Opacity that is typically applied to a UI component in its disabled state, to make it look grayed out.
  // This was moved here from SunConstants because it's needed by LayoutBox.
  // See https://github.com/phetsims/scenery/issues/1153
  DISABLED_OPACITY: 0.45,

  // The name of the color profile used for projector mode
  PROJECTOR_COLOR_PROFILE_NAME: 'projector'
};

scenery.register( 'SceneryConstants', SceneryConstants );
export default SceneryConstants;