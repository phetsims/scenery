// Copyright 2023, University of Colorado Boulder

/**
 * Represents an abstract rendering program, that may be location-varying
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, PolygonalFace, RenderAlpha, RenderBlendCompose, RenderColor, RenderColorSpaceConversion, RenderFilter, RenderImage, RenderLinearBlend, RenderLinearGradient, RenderPath, RenderPathBoolean, RenderProgramNeeds, RenderRadialBlend, RenderRadialGradient, scenery, SerializedRenderAlpha, SerializedRenderBlendCompose, SerializedRenderColor, SerializedRenderColorSpaceConversion, SerializedRenderFilter, SerializedRenderImage, SerializedRenderLinearBlend, SerializedRenderLinearGradient, SerializedRenderPathBoolean, SerializedRenderRadialBlend, SerializedRenderRadialGradient } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default abstract class RenderProgram {
  public abstract getChildren(): RenderProgram[];
  public abstract withChildren( children: RenderProgram[] ): RenderProgram;
  public abstract getName(): string;

  /**
   * Whether this RenderProgram will return an evaluation (regardless of the position) with an empty alpha value (0).
   * If this is true, we can potentially simplify parts of the RenderProgram tree.
   *
   * NOTE: Default implementation, should be overridden by subclasses that have more specific needs
   */
  public isFullyTransparent(): boolean {
    assert && assert( this.getChildren().length > 0, 'Required implementation for leaves' );

    return _.every( this.getChildren(), child => child.isFullyTransparent() );
  }

  /**
   * Whether this RenderProgram will return an evaluation (regardless of the position) with full alpha value (1).
   * If this is true, we can potentially simplify parts of the RenderProgram tree.
   *
   * NOTE: Default implementation, should be overridden by subclasses that have more specific needs
   */
  public isFullyOpaque(): boolean {
    assert && assert( this.getChildren().length > 0, 'Required implementation for leaves' );

    return _.every( this.getChildren(), child => child.isFullyOpaque() );
  }

  /**
   * Whether this RenderProgram will want a computed face for its evaluation
   * If it's not needed, we can give bogus info to the program.
   *
   * NOTE: Default implementation, should be overridden by subclasses that have more specific needs
   */
  public needsFace(): boolean {
    assert && assert( this.getChildren().length > 0, 'Required implementation for leaves' );

    return _.some( this.getChildren(), child => child.needsFace() );
  }

  /**
   * Whether this RenderProgram will want a computed area for its evaluation
   * If it's not needed, we can give bogus info to the program.
   *
   * NOTE: Default implementation, should be overridden by subclasses that have more specific needs
   */
  public needsArea(): boolean {
    assert && assert( this.getChildren().length > 0, 'Required implementation for leaves' );

    return _.some( this.getChildren(), child => child.needsArea() );
  }

  /**
   * Whether this RenderProgram will want a computed centroid for its evaluation
   * If it's not needed, we can give bogus info to the program.
   *
   * NOTE: Default implementation, should be overridden by subclasses that have more specific needs
   */
  public needsCentroid(): boolean {
    assert && assert( this.getChildren().length > 0, 'Required implementation for leaves' );

    return _.some( this.getChildren(), child => child.needsCentroid() );
  }

  // public abstract isFullyTransparent(): boolean;
  // public abstract isFullyOpaque(): boolean;
  //
  //
  // public abstract needsFace(): boolean;
  // public abstract needsArea(): boolean;
  // public abstract needsCentroid(): boolean;

  public abstract simplify( pathTest?: ( renderPath: RenderPath ) => boolean ): RenderProgram;

  // Premultiplied linear RGB, ignoring the path
  public abstract evaluate(
    face: ClippableFace | null, // if null, it is fully covered
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    pathTest?: ( renderPath: RenderPath ) => boolean
  ): Vector4;

  public equals( other: RenderProgram ): boolean {
    return this === other || (
      this.getName() === other.getName() &&
      this.getChildren().length === other.getChildren().length &&
      _.every( this.getChildren(), ( child, i ) => child.equals( other.getChildren()[ i ] ) ) &&
      this.equalsTyped( other as this ) // If they have the same name, should be the same type(!)
    );
  }

  protected equalsTyped( other: this ): boolean {
    return true;
  }

  /**
   * Returns a new RenderProgram with the given transform applied to it.
   *
   * NOTE: Default implementation, should be overridden by subclasses that have positioning information embedded inside
   */
  public transformed( transform: Matrix3 ): RenderProgram {
    return this.withChildren( this.getChildren().map( child => child.transformed( transform ) ) );
  }

  public depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.getChildren().forEach( child => child.depthFirst( callback ) );
    callback( this );
  }

  public replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return this.withChildren( this.getChildren().map( child => child.replace( callback ) ) );
    }
  }

  public getNeeds(): RenderProgramNeeds {
    return new RenderProgramNeeds( this.needsFace(), this.needsArea(), this.needsCentroid() );
  }

  public toRecursiveString( indent: string ): string {
    const extra = this.getExtraDebugString();
    let string = `${indent}${this.getName()}${extra ? ` (${extra})` : ''}`;

    this.getChildren().forEach( child => {
      string += '\n' + child.toRecursiveString( indent + '  ' );
    } );

    return string;
  }

  protected getExtraDebugString(): string {
    return '';
  }

  public abstract serialize(): SerializedRenderProgram;

  public static deserialize( obj: SerializedRenderProgram ): RenderProgram {
    if ( obj.type === 'RenderAlpha' ) {
      return RenderAlpha.deserialize( obj as SerializedRenderAlpha );
    }
    else if ( obj.type === 'RenderBlendCompose' ) {
      return RenderBlendCompose.deserialize( obj as SerializedRenderBlendCompose );
    }
    else if ( obj.type === 'RenderColor' ) {
      return RenderColor.deserialize( obj as SerializedRenderColor );
    }
    else if ( obj.type === 'RenderPathBoolean' ) {
      return RenderPathBoolean.deserialize( obj as SerializedRenderPathBoolean );
    }
    else if ( obj.type === 'RenderFilter' ) {
      return RenderFilter.deserialize( obj as SerializedRenderFilter );
    }
    else if ( obj.type === 'RenderImage' ) {
      return RenderImage.deserialize( obj as SerializedRenderImage );
    }
    else if ( obj.type === 'RenderLinearBlend' ) {
      return RenderLinearBlend.deserialize( obj as SerializedRenderLinearBlend );
    }
    else if ( obj.type === 'RenderLinearGradient' ) {
      return RenderLinearGradient.deserialize( obj as SerializedRenderLinearGradient );
    }
    else if ( obj.type === 'RenderRadialBlend' ) {
      return RenderRadialBlend.deserialize( obj as SerializedRenderRadialBlend );
    }
    else if ( obj.type === 'RenderRadialGradient' ) {
      return RenderRadialGradient.deserialize( obj as SerializedRenderRadialGradient );
    }
    else if ( obj.type === 'RenderColorSpaceConversion' ) {
      return RenderColorSpaceConversion.deserialize( obj as SerializedRenderColorSpaceConversion );
    }

    throw new Error( `Unrecognized RenderProgram type: ${obj.type}` );
  }

  public static ensureFace( face: ClippableFace | null, minX: number, minY: number, maxX: number, maxY: number ): ClippableFace {
    return face || PolygonalFace.fromBoundsValues( minX, minY, maxX, maxY );
  }

  public static ensureCentroid( face: ClippableFace | null, area: number, minX: number, minY: number, maxX: number, maxY: number ): Vector2 {
    return face ? face.getCentroid( area ) : new Vector2( ( minX + maxX ) / 2, ( minY + maxY ) / 2 );
  }
}

scenery.register( 'RenderProgram', RenderProgram );

export type SerializedRenderProgram = {
  type: string;
};
