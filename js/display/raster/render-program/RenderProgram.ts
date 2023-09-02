// Copyright 2023, University of Colorado Boulder

/**
 * Represents an abstract rendering program, that may be location-varying
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, PolygonalFace, RenderableFace, RenderAlpha, RenderBarycentricBlend, RenderBarycentricPerspectiveBlend, RenderBlendCompose, RenderColor, RenderColorSpace, RenderColorSpaceConversion, RenderDepthSort, RenderFilter, RenderImage, RenderLinearBlend, RenderLinearGradient, RenderNormalize, RenderPath, RenderPathBoolean, RenderProgramNeeds, RenderRadialBlend, RenderRadialGradient, RenderStack, scenery, SerializedRenderAlpha, SerializedRenderBarycentricBlend, SerializedRenderBarycentricPerspectiveBlend, SerializedRenderBlendCompose, SerializedRenderColor, SerializedRenderColorSpaceConversion, SerializedRenderDepthSort, SerializedRenderFilter, SerializedRenderImage, SerializedRenderLinearBlend, SerializedRenderLinearGradient, SerializedRenderNormalize, SerializedRenderPathBoolean, SerializedRenderRadialBlend, SerializedRenderRadialGradient, SerializedRenderStack } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default abstract class RenderProgram {
  /**
   * Should return all of the "child" RenderPrograms that this RenderProgram is composed of (the ones it uses to compute
   * a result color).
   */
  public abstract getChildren(): RenderProgram[];

  /**
   * Should return an otherwise-identical version of the RenderProgram with the given children.
   */
  public abstract withChildren( children: RenderProgram[] ): RenderProgram;

  /**
   * Should return the name of the RenderProgram, for serialization and debugging purposes.
   */
  public abstract getName(): string;

  /**
   * Whether this RenderProgram will return an evaluation (regardless of the position) with an empty alpha value (0).
   * If this is true, we can potentially simplified parts of the RenderProgram tree.
   *
   * NOTE: Default implementation, should be overridden by subclasses that have more specific needs
   */
  public isFullyTransparent(): boolean {
    assert && assert( this.getChildren().length > 0, 'Required implementation for leaves' );

    return _.every( this.getChildren(), child => child.isFullyTransparent() );
  }

  /**
   * Whether this RenderProgram will return an evaluation (regardless of the position) with full alpha value (1).
   * If this is true, we can potentially simplified parts of the RenderProgram tree.
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

  public simplified(): RenderProgram {
    let changed = false;
    const children = this.getChildren().map( child => {
      const simplified = child.simplified();
      if ( simplified !== child ) {
        changed = true;
      }
      return simplified;
    } );
    if ( changed ) {
      return this.withChildren( children );
    }
    else {
      return this;
    }
  }

  // Premultiplied linear RGB, ignoring the path
  public abstract evaluate(
    face: ClippableFace | null, // if null, it is fully covered
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
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

  // TODO: add early exit!
  public depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.getChildren().forEach( child => child.depthFirst( callback ) );
    callback( this );
  }

  public containsRenderProgram( renderProgram: RenderProgram ): boolean {
    let result = false;

    this.depthFirst( candidateRenderProgram => {
      if ( candidateRenderProgram === renderProgram ) {
        result = true;
      }

      // TODO: early exit!!!!
    } );

    return result;
  }

  public replace( callback: ( program: RenderProgram ) => RenderProgram | null ): RenderProgram {
    // TODO: preserve DAG!
    const replaced = callback( this );
    if ( replaced ) {
      return replaced;
    }
    else {
      return this.withChildren( this.getChildren().map( child => child.replace( callback ) ) );
    }
  }

  public withPathInclusion( pathTest: ( renderPath: RenderPath ) => boolean ): RenderProgram {
    if ( this instanceof RenderPathBoolean ) {
      if ( pathTest( this.path ) ) {
        return this.inside.withPathInclusion( pathTest );
      }
      else {
        return this.outside.withPathInclusion( pathTest );
      }
    }
    else {
      return this.withChildren( this.getChildren().map( child => child.withPathInclusion( pathTest ) ) );
    }
  }

  public isSplittable(): boolean {
    return false;
  }

  public split( face: RenderableFace ): RenderableFace[] {
    // TODO: should we handle the renderprogram splitting here?
    throw new Error( 'unimplemented' );
  }

  public getNeeds(): RenderProgramNeeds {
    return new RenderProgramNeeds( this.needsFace(), this.needsArea(), this.needsCentroid() );
  }

  public colorConverted( fromSpace: RenderColorSpace, toSpace: RenderColorSpace ): RenderProgram {
    return RenderColorSpaceConversion.convert( this, fromSpace, toSpace );
  }

  public toRecursiveString( indent = '' ): string {
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
    if ( obj.type === 'RenderStack' ) {
      return RenderStack.deserialize( obj as SerializedRenderStack );
    }
    else if ( obj.type === 'RenderColor' ) {
      return RenderColor.deserialize( obj as SerializedRenderColor );
    }
    else if ( obj.type === 'RenderAlpha' ) {
      return RenderAlpha.deserialize( obj as SerializedRenderAlpha );
    }
    else if ( obj.type === 'RenderDepthSort' ) {
      return RenderDepthSort.deserialize( obj as SerializedRenderDepthSort );
    }
    else if ( obj.type === 'RenderBlendCompose' ) {
      return RenderBlendCompose.deserialize( obj as SerializedRenderBlendCompose );
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
    else if ( obj.type === 'RenderNormalize' ) {
      return RenderNormalize.deserialize( obj as SerializedRenderNormalize );
    }
    else if ( obj.type === 'RenderLinearBlend' ) {
      return RenderLinearBlend.deserialize( obj as SerializedRenderLinearBlend );
    }
    else if ( obj.type === 'RenderBarycentricBlend' ) {
      return RenderBarycentricBlend.deserialize( obj as SerializedRenderBarycentricBlend );
    }
    else if ( obj.type === 'RenderBarycentricPerspectiveBlend' ) {
      return RenderBarycentricPerspectiveBlend.deserialize( obj as SerializedRenderBarycentricPerspectiveBlend );
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
}

scenery.register( 'RenderProgram', RenderProgram );

export type SerializedRenderProgram = {
  type: string;
};
