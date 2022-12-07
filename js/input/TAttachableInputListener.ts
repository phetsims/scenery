// Copyright 2022, University of Colorado Boulder

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { TInputListener } from '../imports.js';

type TAttachableInputListener = {
  // Has to be interruptable
  interrupt: () => void;
} & TInputListener;
export default TAttachableInputListener;

