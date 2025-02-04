// Copyright 2025, University of Colorado Boulder

/**
 * Display has an axon `Property to indicate which component is focused (or null if no
 * scenery Node has focus). By passing the tandem and phetioTye, PhET-iO is able to interoperate (save, restore,
 * control, observe what is currently focused). See FocusManager.pdomFocus for setting the focus. Don't set the value
 * of this Property directly.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Property from '../../../axon/js/Property.js';
import type Node from '../nodes/Node.js';
import Focus from './Focus.js';
import Tandem from '../../../tandem/js/Tandem.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';

export const pdomFocusProperty = new Property<Focus | null>( null, {
  tandem: Tandem.GENERAL_MODEL.createTandem( 'pdomFocusProperty' ),
  phetioDocumentation: 'Stores the current focus in the Parallel DOM, null if nothing has focus. This is not updated ' +
                       'based on mouse or touch input, only keyboard and other alternative inputs. Note that this only ' +
                       'applies to simulations that support alternative input.',
  phetioValueType: NullableIO( Focus.FocusIO ),
  phetioState: false,
  phetioFeatured: true,
  phetioReadOnly: true
} );

/**
 * Get the Node that currently has DOM focus, the leaf-most Node of the Focus Trail. Null if no
 * Node has focus.
 */
export const getPDOMFocusedNode = (): Node | null => {
  let focusedNode = null;
  const focus = pdomFocusProperty.value;
  if ( focus ) {
    focusedNode = focus.trail.lastNode();
  }
  return focusedNode;
};