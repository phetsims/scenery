// Copyright 2023, University of Colorado Boulder

/**
 * A logical "press" for the MultiListener, capturing information when a Pointer goes down on the screen.
 */

import { Pointer, scenery, Trail } from '../imports.js';
import Vector2 from '../../../dot/js/Vector2.js';

export default class MultiListenerPress {
  public pointer: Pointer;
  public trail: Trail;
  public interrupted: boolean;

  // down point for the new press, in the global coordinate frame
  public readonly initialPoint: Vector2;

  // point for the new press, in the local coordinate frame of the leaf Node of the Trail
  public localPoint: Vector2 | null;

  public constructor( pointer: Pointer, trail: Trail ) {
    this.pointer = pointer;
    this.trail = trail;
    this.interrupted = false;

    this.initialPoint = pointer.point;

    this.localPoint = null;
    this.recomputeLocalPoint();
  }

  /**
   * Compute the local point for this Press, which is the local point for the leaf Node of this Press's Trail.
   */
  public recomputeLocalPoint(): void {
    this.localPoint = this.trail.globalToLocalPoint( this.pointer.point );
  }

  /**
   * The parent point of this press, relative to the leaf Node of this Press's Trail.
   */
  public get targetPoint(): Vector2 {
    return this.trail.globalToParentPoint( this.pointer.point );
  }
}

scenery.register( 'MultiListenerPress', MultiListenerPress );