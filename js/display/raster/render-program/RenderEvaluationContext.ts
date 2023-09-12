// Copyright 2023, University of Colorado Boulder

/**
 * Represents the state passed through RenderProgram evaluation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, PolygonalFace, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';

export default class RenderEvaluationContext {

  // if null AND we have a need set for a face, it is fully covered
  public face: ClippableFace | null = null;

  // TODO: documentation!
  public area = 0;
  public centroid = new Vector2( 0, 0 );
  public minX = 0;
  public minY = 0;
  public maxX = 0;
  public maxY = 0;

  public set(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): this {
    this.face = face;
    this.area = area;
    this.centroid.set( centroid );
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;

    return this;
  }

  public getFace(): ClippableFace {
    return this.face ? this.face : PolygonalFace.fromBoundsValues( this.minX, this.minY, this.maxX, this.maxY );
  }

  public getBounds(): Bounds2 {
    return new Bounds2( this.minX, this.minY, this.maxX, this.maxY );
  }

  public getCenterX(): number {
    return ( this.minX + this.maxX ) / 2;
  }

  public getCenterY(): number {
    return ( this.minY + this.maxY ) / 2;
  }

  public hasArea(): boolean {
    return isFinite( this.area );
  }

  public hasCentroid(): boolean {
    return this.centroid.isFinite();
  }

  public writeBoundsCentroid( centroid: Vector2 ): Vector2 {
    centroid.setXY(
      ( this.minX + this.maxX ) / 2,
      ( this.minY + this.maxY ) / 2
    );
    return centroid;
  }
}

scenery.register( 'RenderEvaluationContext', RenderEvaluationContext );
