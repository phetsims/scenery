// Copyright 2023, University of Colorado Boulder

/**
 * Represents an element in the RichText hierarchy that has child content (renders nothing on its own, but has its own
 * scale, positioning, style, etc.). <span> or <b> are examples of something that would create this.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Pool, { TPoolable } from '../../../../phet-core/js/Pool.js';
import { Node, RichTextCleanable, RichTextLeaf, RichTextNode, scenery } from '../../imports.js';

export default class RichTextElement extends RichTextCleanable( Node ) implements TPoolable {

  private isLTR!: boolean;

  // The amount of local-coordinate spacing to apply on each side
  public leftSpacing!: number;
  public rightSpacing!: number;

  /**
   * @param isLTR - Whether this container will lay out elements in the left-to-right order (if false, will be
   *                          right-to-left).
   */
  public constructor( isLTR: boolean ) {
    super();

    this.initialize( isLTR );
  }

  public initialize( isLTR: boolean ): this {
    this.isLTR = isLTR;
    this.leftSpacing = 0;
    this.rightSpacing = 0;

    return this;
  }

  /**
   * Adds a child element.
   *
   * @returns- Whether the item was actually added.
   */
  public addElement( element: RichTextElement | RichTextLeaf | RichTextNode ): boolean {

    const hadChild = this.children.length > 0;
    const hasElement = element.width > 0;

    // May be at a different scale, which we need to handle
    const elementScale = element.getScaleVector().x;
    const leftElementSpacing = element.leftSpacing * elementScale;
    const rightElementSpacing = element.rightSpacing * elementScale;

    // If there is nothing, then no spacing should be handled
    if ( !hadChild && !hasElement ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( 'No child or element, ignoring' );
      return false;
    }
    else if ( !hadChild ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `First child, ltr:${this.isLTR}, spacing: ${this.isLTR ? rightElementSpacing : leftElementSpacing}` );
      if ( this.isLTR ) {
        element.left = 0;
        this.leftSpacing = leftElementSpacing;
        this.rightSpacing = rightElementSpacing;
      }
      else {
        element.right = 0;
        this.leftSpacing = leftElementSpacing;
        this.rightSpacing = rightElementSpacing;
      }
      this.addChild( element );
      return true;
    }
    else if ( !hasElement ) {
      sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `No element, adding spacing, ltr:${this.isLTR}, spacing: ${leftElementSpacing + rightElementSpacing}` );
      if ( this.isLTR ) {
        this.rightSpacing += leftElementSpacing + rightElementSpacing;
      }
      else {
        this.leftSpacing += leftElementSpacing + rightElementSpacing;
      }
    }
    else {
      if ( this.isLTR ) {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `LTR add ${this.rightSpacing} + ${leftElementSpacing}` );
        element.left = this.localRight + this.rightSpacing + leftElementSpacing;
        this.rightSpacing = rightElementSpacing;
      }
      else {
        sceneryLog && sceneryLog.RichText && sceneryLog.RichText( `RTL add ${this.leftSpacing} + ${rightElementSpacing}` );
        element.right = this.localLeft - this.leftSpacing - rightElementSpacing;
        this.leftSpacing = leftElementSpacing;
      }
      this.addChild( element );
      return true;
    }
    return false;
  }

  /**
   * Adds an amount of spacing to the "before" side.
   */
  public addExtraBeforeSpacing( amount: number ): void {
    if ( this.isLTR ) {
      this.leftSpacing += amount;
    }
    else {
      this.rightSpacing += amount;
    }
  }

  public freeToPool(): void {
    RichTextElement.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextElement );

}

scenery.register( 'RichTextElement', RichTextElement );
