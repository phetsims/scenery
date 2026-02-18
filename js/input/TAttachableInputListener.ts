// Copyright 2022-2026, University of Colorado Boulder

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import TInputListener from '../input/TInputListener.js';

type TAttachableInputListener = {
  // Has to be interruptable
  interrupt: () => void;
} & TInputListener;
export default TAttachableInputListener;