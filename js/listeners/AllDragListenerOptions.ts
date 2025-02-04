// Copyright 2024-2025, University of Colorado Boulder

/**
 * Options used by many drag listeners in scenery. At this time, that includes DragListener and KeyboardDragListener.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TProperty from '../../../axon/js/TProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import type { SceneryListenerCallback, SceneryListenerNullableCallback } from '../listeners/PressListener.js';

type MapPosition = ( point: Vector2 ) => Vector2;

export type AllDragListenerOptions<Listener, DOMEvent extends Event> = {

  // Called when the drag is started.
  start?: SceneryListenerCallback<Listener, DOMEvent> | null;

  // Called when this listener is dragged.
  drag?: SceneryListenerCallback<Listener, DOMEvent> | null;

  // Called when the drag is ended.
  // NOTE: This will also be called if the drag is ended due to being interrupted or canceled.
  end?: SceneryListenerNullableCallback<Listener, DOMEvent> | null;

  // If provided, it will be synchronized with the drag position in the model coordinate frame. The optional transform
  // is applied.
  positionProperty?: Pick<TProperty<Vector2>, 'value'> | null;

  // If provided, this will be used to convert between the parent (view) and model coordinate frames. Most useful
  // when you also provide a positionProperty.
  transform?: Transform3 | TReadOnlyProperty<Transform3> | null;

  // If provided, the model position will be constrained to these bounds.
  dragBoundsProperty?: TReadOnlyProperty<Bounds2 | null> | null;

  // If provided, this allows custom mapping from the desired position (i.e. where the pointer is, or where the
  // KeyboardDragListener will set the position) to the actual position that will be used.
  mapPosition?: null | MapPosition;

  // If true, the target Node will be translated during the drag operation.
  translateNode?: boolean;
};