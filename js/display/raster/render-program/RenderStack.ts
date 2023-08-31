// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram for repeated compositing of multiple RenderPrograms in a row with normal blending and source-over
 * Porter-Duff composition.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, RenderColor, RenderProgram, scenery, SerializedRenderProgram } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

export default class RenderStack extends RenderProgram {
  /**
   * @param children - Ordered from back to front, like Scenery's Node.children
   */
  public constructor(
    public readonly children: RenderProgram[]
  ) {
    super();
  }

  public override getName(): string {
    return 'RenderStack';
  }

  public override getChildren(): RenderProgram[] {
    return this.children;
  }

  public override withChildren( children: RenderProgram[] ): RenderStack {
    assert && assert( children.length === this.children.length );
    return new RenderStack( children );
  }

  public override simplified(): RenderProgram {
    let children = this.children.map( child => child.simplified() ).filter( child => !child.isFullyTransparent() );

    // If there is an opaque child, nothing below it matters (drop everything before it)
    for ( let i = children.length - 1; i >= 0; i-- ) {
      const child = children[ i ];
      if ( child.isFullyOpaque() ) {
        children = children.slice( i );
        break;
      }
    }

    // Collapse other RenderStacks into this one
    const collapsedChildren: RenderProgram[] = [];
    for ( let i = 0; i < children.length; i++ ) {
      const child = children[ i ];
      if ( child instanceof RenderStack ) {
        collapsedChildren.push( ...child.children );
      }
      else {
        collapsedChildren.push( child );
      }
    }
    children = collapsedChildren;

    // Attempt to blend adjacent colors
    const blendedChildren: RenderProgram[] = [];
    for ( let i = 0; i < children.length; i++ ) {
      const child = children[ i ];
      const lastChild = blendedChildren[ blendedChildren.length - 1 ];

      if ( i > 0 && child instanceof RenderColor && lastChild instanceof RenderColor ) {
        blendedChildren.pop();
        blendedChildren.push( new RenderColor( RenderStack.combine( child.color, lastChild.color ) ) );
      }
      else {
        blendedChildren.push( child );
      }
    }
    children = blendedChildren;

    if ( children.length === 0 ) {
      return RenderColor.TRANSPARENT;
    }
    else if ( children.length === 1 ) {
      return children[ 0 ];
    }
    else if ( this.children.length !== children.length || _.some( this.children, ( child, i ) => child !== children[ i ] ) ) {
      return new RenderStack( children );
    }
    else {
      return this;
    }
  }

  public override isFullyOpaque(): boolean {
    return _.some( this.children, child => child.isFullyOpaque() );
  }

  public static combine( a: Vector4, b: Vector4 ): Vector4 {
    const backgroundAlpha = 1 - a.w;

    return new Vector4(
      a.x + backgroundAlpha * b.x,
      a.y + backgroundAlpha * b.y,
      a.z + backgroundAlpha * b.z,
      a.w + backgroundAlpha * b.w
    );
  }

  public override evaluate(
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): Vector4 {
    // non-stack-based (so no shortcut, but stable memory and simple). Could do in the reverse direction

    const color = Vector4.ZERO.copy(); // we will mutate it

    for ( let i = 0; i < this.children.length; i++ ) {
      const blendColor = this.children[ i ].evaluate( face, area, centroid, minX, minY, maxX, maxY );
      const backgroundAlpha = 1 - blendColor.w;

      // Assume premultiplied
      color.setXYZW(
        blendColor.x + backgroundAlpha * color.x,
        blendColor.y + backgroundAlpha * color.y,
        blendColor.z + backgroundAlpha * color.z,
        blendColor.w + backgroundAlpha * color.w
      );
    }

    return color;
  }

  public override serialize(): SerializedRenderStack {
    return {
      type: 'RenderStack',
      children: this.children.map( child => child.serialize() )
    };
  }

  public static override deserialize( obj: SerializedRenderStack ): RenderStack {
    return new RenderStack( obj.children.map( child => RenderProgram.deserialize( child ) ) );
  }
}

scenery.register( 'RenderStack', RenderStack );

export type SerializedRenderStack = {
  type: 'RenderStack';
  children: SerializedRenderProgram[];
};
