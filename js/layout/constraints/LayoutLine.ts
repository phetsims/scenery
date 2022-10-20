// Copyright 2022, University of Colorado Boulder

/**
 * An internal representation of a row/column for grid/flow handling in constraints (set up for pooling)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';

export default class LayoutLine {

  // A range of sizes along the secondary axis that our cells could take up
  // (scenery-internal)
  public min!: number;
  public max!: number;

  // A range of positions where our align:origin content could go out to (the farthest +/- from 0 that our align:origin
  // nodes go).
  // (scenery-internal)
  public minOrigin!: number;
  public maxOrigin!: number;

  // The line's size (along the secondary axis)
  // (scenery-internal)
  public size!: number;

  // The line's position (along the primary axis)
  // (scenery-internal)
  public position!: number;

  protected initializeLayoutLine(): void {
    this.min = 0;
    this.max = Number.POSITIVE_INFINITY;
    this.minOrigin = Number.POSITIVE_INFINITY;
    this.maxOrigin = Number.NEGATIVE_INFINITY;
    this.size = 0;
    this.position = 0;
  }

  /**
   * Whether there was origin-based content in the layout
   * (scenery-internal)
   */
  public hasOrigin(): boolean {
    return isFinite( this.minOrigin ) && isFinite( this.maxOrigin );
  }
}

scenery.register( 'LayoutLine', LayoutLine );
