// Copyright 2023, University of Colorado Boulder

/**
 * Represents an abstract rendering program, that may be location-varying
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, PolygonalFace, RenderableFace, RenderAlpha, RenderBarycentricBlend, RenderBarycentricPerspectiveBlend, RenderBlendCompose, RenderColor, RenderColorSpace, RenderColorSpaceConversion, RenderDepthSort, RenderDiffuse, RenderFilter, RenderImage, RenderLinearBlend, RenderLinearGradient, RenderNormalDebug, RenderNormalize, RenderPath, RenderPathBoolean, RenderProgramNeeds, RenderRadialBlend, RenderRadialGradient, RenderStack, scenery, SerializedRenderAlpha, SerializedRenderBarycentricBlend, SerializedRenderBarycentricPerspectiveBlend, SerializedRenderBlendCompose, SerializedRenderColor, SerializedRenderColorSpaceConversion, SerializedRenderDepthSort, SerializedRenderDiffuse, SerializedRenderFilter, SerializedRenderImage, SerializedRenderLinearBlend, SerializedRenderLinearGradient, SerializedRenderNormalDebug, SerializedRenderNormalize, SerializedRenderPathBoolean, SerializedRenderRadialBlend, SerializedRenderRadialGradient, SerializedRenderStack } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default abstract class RenderProgram {

  public readonly children: RenderProgram[];

  // Whether it is fully simplified (so simplification steps can be skipped)
  public isSimplified = false;

  // Whether it is fully transparent (so we can skip rendering it)
  public readonly isFullyTransparent: boolean;

  // Whether it is fully opaque (so we could potentially skip rendering other things)
  public readonly isFullyOpaque: boolean;

  // Whether this subtree wants a computed face for its evaluation (If not, can give bogus values for evaluate)
  public readonly needsFace: boolean;

  // Whether this subtree wants a computed area for its evaluation (If not, can give bogus values for evaluate)
  public readonly needsArea: boolean;

  // Whether this subtree wants a computed centroid for its evaluation (If not, can give bogus values for evaluate)
  public readonly needsCentroid: boolean;

  // Whether this subtree contains a RenderPathBoolean
  public readonly hasPathBoolean: boolean;

  public constructor(
    children: RenderProgram[],
    isFullyTransparent: boolean,
    isFullyOpaque: boolean,
    needsFace = false,
    needsArea = false,
    needsCentroid = false
  ) {
    this.children = children;
    this.isFullyTransparent = isFullyTransparent;
    this.isFullyOpaque = isFullyOpaque;

    let hasPathBoolean = this instanceof RenderPathBoolean;

    for ( let i = 0; i < children.length; i++ ) {
      const child = children[ i ];

      needsFace = needsFace || child.needsFace;
      needsArea = needsArea || child.needsArea;
      needsCentroid = needsCentroid || child.needsCentroid;
      hasPathBoolean = hasPathBoolean || child.hasPathBoolean;
    }

    this.needsFace = needsFace;
    this.needsArea = needsArea;
    this.needsCentroid = needsCentroid;
    this.hasPathBoolean = hasPathBoolean;
  }

  /**
   * Should return an otherwise-identical version of the RenderProgram with the given children.
   */
  public abstract withChildren( children: RenderProgram[] ): RenderProgram;

  /**
   * Should return the name of the RenderProgram, for serialization and debugging purposes.
   */
  public abstract getName(): string;

  public simplified(): RenderProgram {
    if ( this.isSimplified ) {
      return this;
    }

    let hasSimplifiedChild = false;
    const children: RenderProgram[] = [];
    for ( let i = 0; i < this.children.length; i++ ) {
      const child = this.children[ i ];
      const simplifiedChild = child.simplified();
      children.push( simplifiedChild );
      hasSimplifiedChild = hasSimplifiedChild || simplifiedChild !== child;
    }

    const potentialSimplified = this.getSimplified( children );
    if ( potentialSimplified ) {
      potentialSimplified.isSimplified = true; // convenience flag, so subtypes don't have to set it
      return potentialSimplified;
    }
    else if ( hasSimplifiedChild ) {
      const result = this.withChildren( children );
      result.isSimplified = true;
      return result;
    }
    else {
      this.isSimplified = true;
      return this;
    }
  }

  protected getSimplified( children: RenderProgram[] ): RenderProgram | null {
    return null;
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
      this.children.length === other.children.length &&
      _.every( this.children, ( child, i ) => child.equals( other.children[ i ] ) ) &&
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
    return this.withChildren( this.children.map( child => child.transformed( transform ) ) );
  }

  // TODO: add early exit!
  public depthFirst( callback: ( program: RenderProgram ) => void ): void {
    this.children.forEach( child => child.depthFirst( callback ) );
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
      return this.withChildren( this.children.map( child => child.replace( callback ) ) );
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
      return this.withChildren( this.children.map( child => child.withPathInclusion( pathTest ) ) );
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
    return new RenderProgramNeeds( this.needsFace, this.needsArea, this.needsCentroid );
  }

  public colorConverted( fromSpace: RenderColorSpace, toSpace: RenderColorSpace ): RenderProgram {
    return RenderColorSpaceConversion.convert( this, fromSpace, toSpace );
  }

  public toRecursiveString( indent = '' ): string {
    const extra = this.getExtraDebugString();
    let string = `${indent}${this.getName()}${extra ? ` (${extra})` : ''}`;

    this.children.forEach( child => {
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
    else if ( obj.type === 'RenderDiffuse' ) {
      return RenderDiffuse.deserialize( obj as SerializedRenderDiffuse );
    }
    else if ( obj.type === 'RenderNormalDebug' ) {
      return RenderNormalDebug.deserialize( obj as SerializedRenderNormalDebug );
    }

    throw new Error( `Unrecognized RenderProgram type: ${obj.type}` );
  }

  public static ensureFace( face: ClippableFace | null, minX: number, minY: number, maxX: number, maxY: number ): ClippableFace {
    return face || PolygonalFace.fromBoundsValues( minX, minY, maxX, maxY );
  }

  public static closureIsFullyTransparent( renderProgram: RenderProgram ): boolean {
    return renderProgram.isFullyTransparent;
  }

  public static closureIsFullyOpaque( renderProgram: RenderProgram ): boolean {
    return renderProgram.isFullyOpaque;
  }

  public static closureSimplified( renderProgram: RenderProgram ): RenderProgram {
    return renderProgram.simplified();
  }
}

scenery.register( 'RenderProgram', RenderProgram );

export type SerializedRenderProgram = {
  type: string;
};
