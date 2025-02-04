// Copyright 2023-2025, University of Colorado Boulder

/**
 * A leaf node element in the RichText, which will display a Node (e.g. with the `nodes` or `tags` feature)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Pool, { TPoolable } from '../../../../phet-core/js/Pool.js';
import Node from '../../nodes/Node.js';
import RichTextCleanable from '../../util/rich-text/RichTextCleanable.js';
import scenery from '../../scenery.js';

export default class RichTextNode extends RichTextCleanable( Node ) implements TPoolable {

  public readonly leftSpacing = 0;
  public readonly rightSpacing = 0;

  public constructor( content: Node ) {
    super();

    this.initialize( content );
  }

  public initialize( content: Node ): this {
    this.addChild( content );

    return this;
  }

  /**
   * Cleans references that could cause memory leaks (as those things may contain other references).
   */
  public override clean(): void {
    super.clean();

    this.removeAllChildren();
  }

  /**
   * Whether this leaf will fit in the specified amount of space
   */
  public fitsIn( widthAvailable: number ): boolean {
    return this.width <= widthAvailable;
  }

  public freeToPool(): void {
    RichTextNode.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextNode );
}

scenery.register( 'RichTextNode', RichTextNode );