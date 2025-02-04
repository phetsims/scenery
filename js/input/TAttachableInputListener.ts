// Copyright 2022-2025, University of Colorado Boulder

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TInputListener from '../input/TInputListener.js';

type TAttachableInputListener = {
  // Has to be interruptable
  interrupt: () => void;
} & TInputListener;
export default TAttachableInputListener;