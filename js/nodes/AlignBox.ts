// Copyright 2016-2022, University of Colorado Boulder

/**
 * A Node that will align child (content) Node within a specific bounding box.
 *
 * If a custom alignBounds is provided, content will be aligned within that bounding box. Otherwise, it will be aligned
 * within a bounding box with the left-top corner of (0,0) of the necessary size to include both the content and
 * all the margins.
 *
 * Preferred sizes will set the alignBounds (to a minimum x/y of 0, and a maximum x/y of preferredWidth/preferredHeight)
 *
 * If alignBounds or a specific preferred size have not been set yet, the AlignBox will not adjust things on that
 * dimension.
 *
 * There are four margins: left, right, top, bottom. They can be set independently, or multiple can be set at the
 * same time (xMargin, yMargin and margin).
 *
 * NOTE: AlignBox resize may not happen immediately, and may be delayed until bounds of a alignBox's child occurs.
 *       layout updates can be forced with invalidateAlignment(). If the alignBox's content that changed is connected
 *       to a Scenery display, its bounds will update when Display.updateDisplay() will called, so this will guarantee
 *       that the layout will be applied before it is displayed. alignBox.getBounds() will not force a refresh, and
 *       may return stale bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import optionize from '../../../phet-core/js/optionize.js';
import { AlignGroup, HeightSizable, HeightSizableNode, HeightSizableSelfOptions, isHeightSizable, isWidthSizable, LayoutConstraint, Node, NodeOptions, scenery, WidthSizable, WidthSizableNode, WidthSizableSelfOptions } from '../imports.js';

const ALIGNMENT_CONTAINER_OPTION_KEYS = [
  'alignBounds', // {Bounds2|null} - See setAlignBounds() for more documentation
  'xAlign', // {string} - 'left', 'center', 'right' or 'stretch', see setXAlign() for more documentation
  'yAlign', // {string} - 'top', 'center', 'bottom' or 'stretch', see setYAlign() for more documentation
  'margin', // {number} - Sets all margins, see setMargin() for more documentation
  'xMargin', // {number} - Sets horizontal margins, see setXMargin() for more documentation
  'yMargin', // {number} - Sets vertical margins, see setYMargin() for more documentation
  'leftMargin', // {number} - Sets left margin, see setLeftMargin() for more documentation
  'rightMargin', // {number} - Sets right margin, see setRightMargin() for more documentation
  'topMargin', // {number} - Sets top margin, see setTopMargin() for more documentation
  'bottomMargin', // {number} - Sets bottom margin, see setBottomMargin() for more documentation
  'group' // {AlignGroup|null} - Share bounds with others, see setGroup() for more documentation
];

export const XAlignValues = [ 'left', 'center', 'right', 'stretch' ] as const;
export type XAlign = ( typeof XAlignValues )[number];

export const YAlignValues = [ 'top', 'center', 'bottom', 'stretch' ] as const;
export type YAlign = ( typeof YAlignValues )[number];

type SelfOptions = {
  alignBounds?: Bounds2 | null;
  xAlign?: XAlign;
  yAlign?: YAlign;
  margin?: number;
  xMargin?: number;
  yMargin?: number;
  leftMargin?: number;
  rightMargin?: number;
  topMargin?: number;
  bottomMargin?: number;
  group?: AlignGroup | null;
  canShrink?: boolean;
};

export type AlignBoxOptions = SelfOptions & Omit<NodeOptions, 'children'> & WidthSizableSelfOptions & HeightSizableSelfOptions;

export default class AlignBox extends WidthSizable( HeightSizable( Node ) ) {

  // Our actual content
  private _content: Node;

  // Controls the bounds in which content is aligned.
  private _alignBounds: Bounds2 | null;

  // Whether x/y has been set
  private _xSet = false;
  private _ySet = false;

  // How to align the content when the alignBounds are larger than our content with its margins.
  private _xAlign: XAlign;
  private _yAlign: YAlign;

  // How much space should be on each side.
  private _leftMargin: number;
  private _rightMargin: number;
  private _topMargin: number;
  private _bottomMargin: number;

  // If available, an AlignGroup that will control our alignBounds
  private _group: AlignGroup | null;

  private readonly constraint: AlignBoxConstraint;

  // Callback for when bounds change (takes no arguments)
  _contentBoundsListener = () => {};

  /**
   * An individual container for an alignment group. Will maintain its size to match that of the group by overriding
   * its localBounds, and will position its content inside its localBounds by respecting its alignment and margins.
   *
   * @param content - Content to align inside of the alignBox
   * @param [providedOptions] - AlignBox-specific options are documented in ALIGNMENT_CONTAINER_OPTION_KEYS
   *                    above, and can be provided along-side options for Node
   */
  constructor( content: Node, providedOptions?: AlignBoxOptions ) {

    const options = optionize<AlignBoxOptions, Pick<SelfOptions, 'canShrink'>, NodeOptions>()( {
      children: [ content ],
      canShrink: false
    }, providedOptions );

    super();

    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    this._content = content;
    this._alignBounds = null;

    this._xAlign = 'center';
    this._yAlign = 'center';
    this._leftMargin = 0;
    this._rightMargin = 0;
    this._topMargin = 0;
    this._bottomMargin = 0;
    this._group = null;
    this._contentBoundsListener = this.invalidateAlignment.bind( this );

    this.localBounds = new Bounds2( 0, 0, 0, 0 );

    this.constraint = new AlignBoxConstraint( this, content, options.canShrink );

    // Will be removed by dispose()
    this._content.boundsProperty.link( this._contentBoundsListener );

    this.mutate( options );

    // Update alignBounds based on preferred sizes
    Property.multilink( [ this.preferredWidthProperty, this.preferredHeightProperty ], ( preferredWidth: number | null, preferredHeight: number | null ) => {
      if ( preferredWidth !== null || preferredHeight !== null ) {
        const bounds = this._alignBounds || new Bounds2( 0, 0, 0, 0 );

        // Overwrite bounds with any preferred setting, with the left/top at 0
        if ( preferredWidth ) {
          bounds.minX = 0;
          bounds.maxX = preferredWidth;
          this._xSet = true;
        }
        if ( preferredHeight ) {
          bounds.minY = 0;
          bounds.maxY = preferredHeight;
          this._ySet = true;
        }

        // Manual update and layout
        this._alignBounds = bounds;
        this.constraint.updateLayout();
      }
    } );
  }

  /**
   * Triggers recomputation of the alignment. Should be called if it needs to be refreshed.
   *
   * NOTE: alignBox.getBounds() will not trigger a bounds validation for our content, and thus WILL NOT trigger
   * layout. content.getBounds() should trigger it, but invalidateAligment() is the preferred method for forcing a
   * re-check.
   */
  invalidateAlignment(): void {
    sceneryLog && sceneryLog.AlignBox && sceneryLog.AlignBox( `AlignBox#${this.id} invalidateAlignment` );
    sceneryLog && sceneryLog.AlignBox && sceneryLog.push();

    // The group update will change our alignBounds if required.
    if ( this._group ) {
      this._group.onAlignBoxResized( this );
    }

    // If the alignBounds didn't change, we'll still need to update our own layout
    this.constraint.updateLayout();

    sceneryLog && sceneryLog.AlignBox && sceneryLog.pop();
  }

  /**
   * Sets the alignment bounds (the bounds in which our content will be aligned). If null, AlignBox will act
   * as if the alignment bounds have a left-top corner of (0,0) and with a width/height that fits the content and
   * bounds.
   *
   * NOTE: If the group is a valid AlignGroup, it will be responsible for setting the alignBounds.
   */
  setAlignBounds( alignBounds: Bounds2 | null ): this {
    assert && assert( alignBounds === null || ( alignBounds instanceof Bounds2 && !alignBounds.isEmpty() && alignBounds.isFinite() ),
      'alignBounds should be a non-empty finite Bounds2' );

    this._xSet = true;
    this._ySet = true;

    // See if the bounds have changed. If both are Bounds2 with the same value, we won't update it.
    if ( this._alignBounds !== alignBounds &&
         ( !alignBounds ||
           !this._alignBounds ||
           !alignBounds.equals( this._alignBounds ) ) ) {
      this._alignBounds = alignBounds;

      this.constraint.updateLayout();
    }

    return this;
  }

  set alignBounds( value: Bounds2 | null ) { this.setAlignBounds( value ); }

  /**
   * Returns the current alignment bounds (if available, see setAlignBounds for details).
   */
  getAlignBounds(): Bounds2 | null {
    return this._alignBounds;
  }

  get alignBounds(): Bounds2 | null { return this.getAlignBounds(); }

  /**
   * Sets the attachment to an AlignGroup. When attached, our alignBounds will be controlled by the group.
   */
  setGroup( group: AlignGroup | null ): this {
    assert && assert( group === null || group instanceof AlignGroup, 'group should be an AlignGroup' );

    if ( this._group !== group ) {
      // Remove from a previous group
      if ( this._group ) {
        this._group.removeAlignBox( this );
      }

      this._group = group;

      // Add to a new group
      if ( this._group ) {
        this._group.addAlignBox( this );
      }
    }

    return this;
  }

  set group( value: AlignGroup | null ) { this.setGroup( value ); }

  /**
   * Returns the attached alignment group (if one exists), or null otherwise.
   */
  getGroup(): AlignGroup | null {
    return this._group;
  }

  get group(): AlignGroup | null { return this.getGroup(); }

  /**
   * Sets the horizontal alignment of this box.
   */
  setXAlign( xAlign: XAlign ): this {
    assert && assert( XAlignValues.includes( xAlign ), `xAlign should be one of: ${XAlignValues}` );

    if ( this._xAlign !== xAlign ) {
      this._xAlign = xAlign;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set xAlign( value: XAlign ) { this.setXAlign( value ); }

  /**
   * Returns the current horizontal alignment of this box.
   */
  getXAlign(): XAlign {
    return this._xAlign;
  }

  get xAlign(): XAlign { return this.getXAlign(); }

  /**
   * Sets the vertical alignment of this box.
   */
  setYAlign( yAlign: YAlign ): this {
    assert && assert( YAlignValues.includes( yAlign ), `xAlign should be one of: ${YAlignValues}` );

    if ( this._yAlign !== yAlign ) {
      this._yAlign = yAlign;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set yAlign( value: YAlign ) { this.setYAlign( value ); }

  /**
   * Returns the current vertical alignment of this box.
   */
  getYAlign(): YAlign {
    return this._yAlign;
  }

  get yAlign(): YAlign { return this.getYAlign(); }

  /**
   * Sets the margin of this box (setting margin values for all sides at once).
   *
   * This margin is the minimum amount of horizontal space that will exist between the content the sides of this
   * box.
   */
  setMargin( margin: number ): this {
    assert && assert( typeof margin === 'number' && isFinite( margin ) && margin >= 0,
      'margin should be a finite non-negative number' );

    if ( this._leftMargin !== margin ||
         this._rightMargin !== margin ||
         this._topMargin !== margin ||
         this._bottomMargin !== margin ) {
      this._leftMargin = this._rightMargin = this._topMargin = this._bottomMargin = margin;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set margin( value: number ) { this.setMargin( value ); }

  /**
   * Returns the current margin of this box (assuming all margin values are the same).
   */
  getMargin(): number {
    assert && assert( this._leftMargin === this._rightMargin &&
    this._leftMargin === this._topMargin &&
    this._leftMargin === this._bottomMargin,
      'Getting margin does not have a unique result if the left and right margins are different' );
    return this._leftMargin;
  }

  get margin(): number { return this.getMargin(); }

  /**
   * Sets the horizontal margin of this box (setting both left and right margins at once).
   *
   * This margin is the minimum amount of horizontal space that will exist between the content and the left and
   * right sides of this box.
   */
  setXMargin( xMargin: number ): this {
    assert && assert( typeof xMargin === 'number' && isFinite( xMargin ) && xMargin >= 0,
      'xMargin should be a finite non-negative number' );

    if ( this._leftMargin !== xMargin || this._rightMargin !== xMargin ) {
      this._leftMargin = this._rightMargin = xMargin;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set xMargin( value: number ) { this.setXMargin( value ); }

  /**
   * Returns the current horizontal margin of this box (assuming the left and right margins are the same).
   */
  getXMargin(): number {
    assert && assert( this._leftMargin === this._rightMargin,
      'Getting xMargin does not have a unique result if the left and right margins are different' );
    return this._leftMargin;
  }

  get xMargin(): number { return this.getXMargin(); }

  /**
   * Sets the vertical margin of this box (setting both top and bottom margins at once).
   *
   * This margin is the minimum amount of vertical space that will exist between the content and the top and
   * bottom sides of this box.
   */
  setYMargin( yMargin: number ): this {
    assert && assert( typeof yMargin === 'number' && isFinite( yMargin ) && yMargin >= 0,
      'yMargin should be a finite non-negative number' );

    if ( this._topMargin !== yMargin || this._bottomMargin !== yMargin ) {
      this._topMargin = this._bottomMargin = yMargin;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set yMargin( value: number ) { this.setYMargin( value ); }

  /**
   * Returns the current vertical margin of this box (assuming the top and bottom margins are the same).
   */
  getYMargin(): number {
    assert && assert( this._topMargin === this._bottomMargin,
      'Getting yMargin does not have a unique result if the top and bottom margins are different' );
    return this._topMargin;
  }

  get yMargin(): number { return this.getYMargin(); }

  /**
   * Sets the left margin of this box.
   *
   * This margin is the minimum amount of horizontal space that will exist between the content and the left side of
   * the box.
   */
  setLeftMargin( leftMargin: number ): this {
    assert && assert( typeof leftMargin === 'number' && isFinite( leftMargin ) && leftMargin >= 0,
      'leftMargin should be a finite non-negative number' );

    if ( this._leftMargin !== leftMargin ) {
      this._leftMargin = leftMargin;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set leftMargin( value: number ) { this.setLeftMargin( value ); }

  /**
   * Returns the current left margin of this box.
   */
  getLeftMargin(): number {
    return this._leftMargin;
  }

  get leftMargin(): number { return this.getLeftMargin(); }

  /**
   * Sets the right margin of this box.
   *
   * This margin is the minimum amount of horizontal space that will exist between the content and the right side of
   * the container.
   */
  setRightMargin( rightMargin: number ): this {
    assert && assert( typeof rightMargin === 'number' && isFinite( rightMargin ) && rightMargin >= 0,
      'rightMargin should be a finite non-negative number' );

    if ( this._rightMargin !== rightMargin ) {
      this._rightMargin = rightMargin;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set rightMargin( value: number ) { this.setRightMargin( value ); }

  /**
   * Returns the current right margin of this box.
   */
  getRightMargin(): number {
    return this._rightMargin;
  }

  get rightMargin(): number { return this.getRightMargin(); }

  /**
   * Sets the top margin of this box.
   *
   * This margin is the minimum amount of vertical space that will exist between the content and the top side of the
   * container.
   */
  setTopMargin( topMargin: number ): this {
    assert && assert( typeof topMargin === 'number' && isFinite( topMargin ) && topMargin >= 0,
      'topMargin should be a finite non-negative number' );

    if ( this._topMargin !== topMargin ) {
      this._topMargin = topMargin;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set topMargin( value: number ) { this.setTopMargin( value ); }

  /**
   * Returns the current top margin of this box.
   */
  getTopMargin(): number {
    return this._topMargin;
  }

  get topMargin(): number { return this.getTopMargin(); }

  /**
   * Sets the bottom margin of this box.
   *
   * This margin is the minimum amount of vertical space that will exist between the content and the bottom side of the
   * container.
   */
  setBottomMargin( bottomMargin: number ): this {
    assert && assert( typeof bottomMargin === 'number' && isFinite( bottomMargin ) && bottomMargin >= 0,
      'bottomMargin should be a finite non-negative number' );

    if ( this._bottomMargin !== bottomMargin ) {
      this._bottomMargin = bottomMargin;

      // Trigger re-layout
      this.invalidateAlignment();
    }

    return this;
  }

  set bottomMargin( value: number ) { this.setBottomMargin( value ); }

  /**
   * Returns the current bottom margin of this box.
   */
  getBottomMargin(): number {
    return this._bottomMargin;
  }

  get bottomMargin(): number { return this.getBottomMargin(); }

  /**
   * Returns the bounding box of this box's content. This will include any margins.
   */
  getContentBounds(): Bounds2 {
    sceneryLog && sceneryLog.AlignBox && sceneryLog.AlignBox( `AlignBox#${this.id} getContentBounds` );
    sceneryLog && sceneryLog.AlignBox && sceneryLog.push();

    const bounds = this._content.bounds;

    sceneryLog && sceneryLog.AlignBox && sceneryLog.pop();

    return new Bounds2( bounds.left - this._leftMargin,
      bounds.top - this._topMargin,
      bounds.right + this._rightMargin,
      bounds.bottom + this._bottomMargin );
  }

  // scenery-internal, designed so that we can ignore adjusting certain dimensions
  setAdjustedLocalBounds( bounds: Bounds2 ): void {
    if ( this._xSet && this._ySet ) {
      this.localBounds = bounds;
    }
    else if ( this._xSet ) {
      const contentBounds = this.getContentBounds();

      this.localBounds = new Bounds2( bounds.minX, contentBounds.minY, bounds.maxX, contentBounds.maxY );
    }
    else if ( this._ySet ) {
      const contentBounds = this.getContentBounds();

      this.localBounds = new Bounds2( contentBounds.minX, bounds.minY, contentBounds.maxX, bounds.maxY );
    }
    else {
      this.localBounds = this.getContentBounds();
    }
  }

  /**
   * Disposes this box, releasing listeners and any references to an AlignGroup
   */
  override dispose(): void {
    // Remove our listener
    this._content.boundsProperty.unlink( this._contentBoundsListener );

    // Disconnects from the group
    this.group = null;

    this.constraint.dispose();

    super.dispose();
  }
}

// Layout logic for AlignBox
class AlignBoxConstraint extends LayoutConstraint {

  private readonly alignBox: AlignBox;
  private readonly content: Node;
  private readonly canShrink: boolean;

  constructor( alignBox: AlignBox, content: Node, canShrink: boolean ) {
    super( alignBox );

    this.alignBox = alignBox;
    this.content = content;
    this.canShrink = canShrink;

    this.addNode( content );
  }

  /**
   * Conditionally updates a certain property of our content's positioning.
   *
   * Essentially does the following (but prevents infinite loops by not applying changes if the numbers are very
   * similar):
   * this._content[ propName ] = this.localBounds[ propName ] + offset;
   *
   * @param propName - A positional property on both Node and Bounds2, e.g. 'left'
   * @param offset - Offset to be applied to the localBounds location.
   */
  private updateProperty( propName: 'left' | 'right' | 'top' | 'bottom' | 'centerX' | 'centerY', offset: number ): void {
    const currentValue = this.content[ propName ];
    const newValue = this.alignBox.localBounds[ propName ] + offset;

    // Prevent infinite loops or stack overflows by ignoring tiny changes
    if ( Math.abs( currentValue - newValue ) > 1e-5 ) {
      this.content[ propName ] = newValue;
    }
  }

  protected override layout(): void {
    super.layout();

    const box = this.alignBox;
    const content = this.content;

    sceneryLog && sceneryLog.AlignBox && sceneryLog.AlignBox( `AlignBoxConstraint#${this.alignBox.id} layout` );
    sceneryLog && sceneryLog.AlignBox && sceneryLog.push();

    if ( !content.bounds.isValid() ) {
      return;
    }

    // If we have alignBounds, use that.
    if ( box.alignBounds !== null ) {
      box.setAdjustedLocalBounds( box.alignBounds );
    }
    // Otherwise, we'll grab a Bounds2 anchored at the upper-left with our required dimensions.
    else {
      const widthWithMargin = box.leftMargin + content.width + box.rightMargin;
      const heightWithMargin = box.topMargin + content.height + box.bottomMargin;
      box.setAdjustedLocalBounds( new Bounds2( 0, 0, widthWithMargin, heightWithMargin ) );
    }

    const minimumWidth = isFinite( content.width ) ? ( isWidthSizable( content ) ? content.minimumWidth || 0 : content.width ) : null;
    const minimumHeight = isFinite( content.height ) ? ( isHeightSizable( content ) ? content.minimumHeight || 0 : content.height ) : null;

    // Don't try to lay out empty bounds
    if ( !content.localBounds.isEmpty() ) {

      if ( box.xAlign === 'center' ) {
        this.updateProperty( 'centerX', ( box.leftMargin - box.rightMargin ) / 2 );
      }
      else if ( box.xAlign === 'left' ) {
        this.updateProperty( 'left', box.leftMargin );
      }
      else if ( box.xAlign === 'right' ) {
        this.updateProperty( 'right', -box.rightMargin );
      }
      else if ( box.xAlign === 'stretch' ) {
        assert && assert( isWidthSizable( content ), 'xAlign:stretch can only be used if WidthSizable is mixed into the content' );
        ( content as WidthSizableNode ).preferredWidth = box.localBounds.width - box.leftMargin - box.rightMargin;
        this.updateProperty( 'left', box.leftMargin );
      }
      else {
        assert && assert( `Bad xAlign: ${box.xAlign}` );
      }

      if ( box.yAlign === 'center' ) {
        this.updateProperty( 'centerY', ( box.topMargin - box.bottomMargin ) / 2 );
      }
      else if ( box.yAlign === 'top' ) {
        this.updateProperty( 'top', box.topMargin );
      }
      else if ( box.yAlign === 'bottom' ) {
        this.updateProperty( 'bottom', -box.bottomMargin );
      }
      else if ( box.yAlign === 'stretch' ) {
        assert && assert( isHeightSizable( content ), 'yAlign:stretch can only be used if HeightSizable is mixed into the content' );
        ( content as HeightSizableNode ).preferredHeight = box.localBounds.height - box.topMargin - box.bottomMargin;
        this.updateProperty( 'top', box.topMargin );
      }
      else {
        assert && assert( `Bad yAlign: ${box.yAlign}` );
      }
    }

    sceneryLog && sceneryLog.AlignBox && sceneryLog.pop();

    // After the layout lock on purpose (we want these to be reentrant, especially if they change) - however only apply
    // this concept if we're capable of shrinking (we want the default to continue to block off the layoutBounds)
    box.minimumWidth = this.canShrink ? minimumWidth : box.localBounds.width;
    box.minimumHeight = this.canShrink ? minimumHeight : box.localBounds.height;
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
AlignBox.prototype._mutatorKeys = ALIGNMENT_CONTAINER_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'AlignBox', AlignBox );
