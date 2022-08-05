// Copyright 2022, University of Colorado Boulder

/**
 * The main type interface for input listeners
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { IInputListener } from '../imports.js';

type IAttachableInputListener = {
  // Has to be interruptable
  interrupt: () => void;
} & IInputListener;
export default IAttachableInputListener // eslint-disable-line
