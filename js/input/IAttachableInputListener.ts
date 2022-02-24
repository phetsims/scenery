// Copyright 2022, University of Colorado Boulder

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IInputListener } from '../imports.js';

interface IAttachableInputListener extends IInputListener {
  // Has to be interruptable
  interrupt: () => void;
}

export default IAttachableInputListener;
