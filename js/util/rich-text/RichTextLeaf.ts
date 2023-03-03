// Copyright 2023, University of Colorado Boulder

/**
 * A leaf (text) element in the RichText, which will display a snippet of Text.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Pool, { TPoolable } from '../../../../phet-core/js/Pool.js';
import { Font, RichText, RichTextCleanable, RichTextUtils, scenery, Text, TextBoundsMethod, TPaint } from '../../imports.js';

export default class RichTextLeaf extends RichTextCleanable( Text ) implements TPoolable {

  public leftSpacing!: number;
  public rightSpacing!: number;

  public constructor( content: string, isLTR: boolean, font: Font | string, boundsMethod: TextBoundsMethod, fill: TPaint, stroke: TPaint, lineWidth: number ) {
    super( '' );

    this.initialize( content, isLTR, font, boundsMethod, fill, stroke, lineWidth );
  }

  public initialize( content: string, isLTR: boolean, font: Font | string, boundsMethod: TextBoundsMethod, fill: TPaint, stroke: TPaint, lineWidth: number ): this {

    // Grab all spaces at the (logical) start
    let whitespaceBefore = '';
    while ( content.startsWith( ' ' ) ) {
      whitespaceBefore += ' ';
      content = content.slice( 1 );
    }

    // Grab all spaces at the (logical) end
    let whitespaceAfter = '';
    while ( content.endsWith( ' ' ) ) {
      whitespaceAfter = ' ';
      content = content.slice( 0, content.length - 1 );
    }

    this.string = RichText.contentToString( content, isLTR );
    this.boundsMethod = boundsMethod;
    this.font = font;
    this.fill = fill;
    this.stroke = stroke;
    this.lineWidth = lineWidth;

    const spacingBefore = whitespaceBefore.length ? RichTextUtils.scratchText.setString( whitespaceBefore ).setFont( font ).width : 0;
    const spacingAfter = whitespaceAfter.length ? RichTextUtils.scratchText.setString( whitespaceAfter ).setFont( font ).width : 0;

    // Turn logical spacing into directional
    this.leftSpacing = isLTR ? spacingBefore : spacingAfter;
    this.rightSpacing = isLTR ? spacingAfter : spacingBefore;

    return this;
  }

  /**
   * Cleans references that could cause memory leaks (as those things may contain other references).
   */
  public override clean(): void {
    super.clean();

    this.fill = null;
    this.stroke = null;
  }

  /**
   * Whether this leaf will fit in the specified amount of space (including, if required, the amount of spacing on
   * the side).
   */
  public fitsIn( widthAvailable: number, hasAddedLeafToLine: boolean, isContainerLTR: boolean ): boolean {
    return this.width + ( hasAddedLeafToLine ? ( isContainerLTR ? this.leftSpacing : this.rightSpacing ) : 0 ) <= widthAvailable;
  }

  public freeToPool(): void {
    RichTextLeaf.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextLeaf );
}

scenery.register( 'RichTextLeaf', RichTextLeaf );
