// Copyright 2021-2022, University of Colorado Boulder

/**
 * SceneryConstants is a collection of constants used throughout scenery.
 *
 * @author Chris Malley (PixelZoom, Inc.)
 */

import { scenery } from './imports.js';

const SceneryConstants = {

  // Opacity that is typically applied to a UI component in its disabled state, to make it look grayed out.
  // This was moved here from SunConstants because it's needed by FlowBox.
  // See https://github.com/phetsims/scenery/issues/1153
  DISABLED_OPACITY: 0.45,

  // The name of the color profile used by default.
  // NOTE: Duplicated in initialize-globals.js.  Duplicated because scenery doesn't include initialize-globals in its
  // standalone build.
  DEFAULT_COLOR_PROFILE: 'default',

  // The name of the color profile used for projector mode
  PROJECTOR_COLOR_PROFILE: 'projector'
};

scenery.register( 'SceneryConstants', SceneryConstants );
export default SceneryConstants;