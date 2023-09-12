// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for alpha (an opacity) applied to a RenderProgram
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { isWindingIncluded, LinearEdge, RenderColor, RenderEvaluationContext, RenderPath, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import { SerializedRenderPath } from './RenderPath.js';

export default class RenderPathBoolean extends RenderProgram {
  public constructor(
    public readonly path: RenderPath,
    public readonly inside: RenderProgram,
    public readonly outside: RenderProgram
  ) {
    super(
      [ inside, outside ],
      inside.isFullyTransparent && outside.isFullyTransparent,
      inside.isFullyOpaque && outside.isFullyOpaque,
      false,
      false,
      true // We'll use the centroid as the point for determining whether we are on the interior of our path
    );
  }

  public override getName(): string {
    return 'RenderPathBoolean';
  }

  public override withChildren( children: RenderProgram[] ): RenderPathBoolean {
    assert && assert( children.length === 2 );
    return new RenderPathBoolean( this.path, children[ 0 ], children[ 1 ] );
  }

  public override transformed( transform: Matrix3 ): RenderProgram {
    return new RenderPathBoolean( this.path.transformed( transform ), this.inside.transformed( transform ), this.outside.transformed( transform ) );
  }

  protected override equalsTyped( other: this ): boolean {
    return this.path === other.path;
  }

  public isOneSided(): boolean {
    return this.outside.isFullyTransparent || this.inside.isFullyTransparent;
  }

  public getOneSide(): RenderProgram {
    assert && assert( this.isOneSided() );

    return this.outside.isFullyTransparent ? this.inside : this.outside;
  }

  public withOneSide( program: RenderProgram ): RenderProgram {
    assert && assert( this.isOneSided() );

    return this.outside.isFullyTransparent ?
           new RenderPathBoolean( this.path, program, this.outside ) :
           new RenderPathBoolean( this.path, this.inside, program );
  }

  public override getSimplified( children: RenderProgram[] ): RenderProgram | null {
    const inside = children[ 0 ];
    const outside = children[ 1 ];

    // TODO: a check to see if the RenderPath is effectively empty?

    if ( inside.isFullyTransparent && outside.isFullyTransparent ) {
      return RenderColor.TRANSPARENT;
    }
    else if ( inside.equals( outside ) ) {
      return inside;
    }
    else {
      return null;
    }
  }

  public override evaluate( context: RenderEvaluationContext ): Vector4 {
    assert && assert( context.hasCentroid() );

    // TODO: ACTUALLY, we should clip the face with our path....
    const windingNumber = LinearEdge.getWindingNumberPolygons( this.path.subpaths, context.centroid );
    const included = isWindingIncluded( windingNumber, this.path.fillRule );

    return ( included ? this.inside : this.outside ).evaluate( context );
  }

  protected override getExtraDebugString(): string {
    return `${this.path.id}`;
  }

  public override serialize(): SerializedRenderPathBoolean {
    return {
      type: 'RenderPathBoolean',
      path: this.path.serialize(),
      inside: this.inside.serialize(),
      outside: this.outside.serialize()
    };
  }

  public static override deserialize( obj: SerializedRenderPathBoolean ): RenderPathBoolean {
    return new RenderPathBoolean( RenderPath.deserialize( obj.path ), RenderProgram.deserialize( obj.inside ), RenderProgram.deserialize( obj.outside ) );
  }

  public static fromInside( path: RenderPath, inside: RenderProgram ): RenderPathBoolean {
    return new RenderPathBoolean( path, inside, RenderColor.TRANSPARENT );
  }
}

scenery.register( 'RenderPathBoolean', RenderPathBoolean );

export type SerializedRenderPathBoolean = {
  type: 'RenderPathBoolean';
  path: SerializedRenderPath;
  inside: SerializedRenderProgram;
  outside: SerializedRenderProgram;
};
