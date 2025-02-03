// Copyright 2025, University of Colorado Boulder

/**
 * Isolated globals for Display, to avoid circular dependencies.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import TEmitter from '../../../axon/js/TEmitter.js';
import TInputListener from '../input/TInputListener.js';
import Emitter from '../../../axon/js/Emitter.js';

export default class DisplayGlobals {
  // Fires when we detect an input event that would be considered a "user gesture" by Chrome, so
  // that we can trigger browser actions that are only allowed as a result.
  // See https://github.com/phetsims/scenery/issues/802 and https://github.com/phetsims/vibe/issues/32 for more
  // information.
  public static userGestureEmitter: TEmitter = new Emitter();

  // Listeners that will be called for every event on ANY Display, see
  // https://github.com/phetsims/scenery/issues/1149. Do not directly modify this!
  public static inputListeners: TInputListener[] = [];
}
scenery.register( 'DisplayGlobals', DisplayGlobals );