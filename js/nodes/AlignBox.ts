// Copyright 2016-2021, University of Colorado Boulder

/**
 * A Node that will align child (content) Node within a specific bounding box.
 *
 * If a custom alignBounds is provided, content will be aligned within that bounding box. Otherwise, it will be aligned
 * within a bounding box with the left-top corner of (0,0) of the necessary size to include both the content and
 * all of the margins.
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

import Bounds2 from '../../../dot/js/Bounds2.js';
import merge from '../../../phet-core/js/merge.js';
import { scenery, Node, AlignGroup, NodeOptions } from '../imports.js';

const ALIGNMENT_CONTAINER_OPTION_KEYS = [
  'alignBounds', // {Bounds2|null} - See setAlignBounds() for more documentation
  'xAlign', // {string} - 'left', 'center', or 'right', see setXAlign() for more documentation
  'yAlign', // {string} - 'top', 'center', or 'bottom', see setYAlign() for more documentation
  'margin', // {number} - Sets all margins, see setMargin() for more documentation
  'xMargin', // {number} - Sets horizontal margins, see setXMargin() for more documentation
  'yMargin', // {number} - Sets vertical margins, see setYMargin() for more documentation
  'leftMargin', // {number} - Sets left margin, see setLeftMargin() for more documentation
  'rightMargin', // {number} - Sets right margin, see setRightMargin() for more documentation
  'topMargin', // {number} - Sets top margin, see setTopMargin() for more documentation
  'bottomMargin', // {number} - Sets bottom margin, see setBottomMargin() for more documentation
  'group' // {AlignGroup|null} - Share bounds with others, see setGroup() for more documentation
];

type XAlign = 'left' | 'center' | 'right';
type YAlign = 'top' | 'center' | 'bottom';

type AlignBoxSelfOptions = {
  alignBounds?: Bounds2 | null,
  xAlign?: XAlign,
  yAlign?: YAlign,
  margin?: number,
  xMargin?: number,
  yMargin?: number,
  leftMargin?: number,
  rightMargin?: number,
  topMargin?: number,
  bottomMargin?: number,
  'group'?: AlignGroup | null,
};

type AlignBoxOptions = AlignBoxSelfOptions & NodeOptions

class AlignBox extends Node {

  // Our actual content
  private _content: Node;

  // Controls the bounds in which content is aligned.
  private _alignBounds: Bounds2 | null;

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

  // Callback for when bounds change (takes no arguments)
  _contentBoundsListener = () => {};

  // Used to prevent loops
  private _layoutLock: boolean;

  /**
   * An individual container for an alignment group. Will maintain its size to match that of the group by overriding
   * its localBounds, and will position its content inside its localBounds by respecting its alignment and margins.
   * @public
   *
   * @param content - Content to align inside of the alignBox
   * @param [options] - AlignBox-specific options are documented in ALIGNMENT_CONTAINER_OPTION_KEYS
   *                    above, and can be provided along-side options for Node
   */
  constructor( content: Node, options?: AlignBoxOptions ) {

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
    this._layoutLock = false;

    // Will be removed by dispose()
    this._content.boundsProperty.lazyLink( this._contentBoundsListener );

    this.mutate( merge( {}, options, {
      children: [ this._content ]
    } ) );
  }

  /**
   * Triggers recomputation of the alignment. Should be called if it needs to be refreshed.
   *
   * NOTE: alignBox.getBounds() will not trigger a bounds validation for our content, and thus WILL NOT trigger
   * layout. content.getBounds() should trigger it, but invalidateAligment() is the preferred method for forcing a
   * re-check.
   */
  invalidateAlignment() {
    sceneryLog && sceneryLog.AlignBox && sceneryLog.AlignBox( `AlignBox#${this.id} invalidateAlignment` );
    sceneryLog && sceneryLog.AlignBox && sceneryLog.push();

    // The group update will change our alignBounds if required.
    if ( this._group ) {
      this._group.onAlignBoxResized( this );
    }

    // If the alignBounds didn't change, we'll still need to update our own layout
    this.updateLayout();

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

    // See if the bounds have changed. If both are Bounds2 with the same value, we won't update it.
    if ( this._alignBounds !== alignBounds &&
         ( !alignBounds ||
           !this._alignBounds ||
           !alignBounds.equals( this._alignBounds ) ) ) {
      this._alignBounds = alignBounds;

      this.updateLayout();
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
    assert && assert( xAlign === 'left' || xAlign === 'center' || xAlign === 'right',
      'xAlign should be one of: \'left\', \'center\', or \'right\'' );

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
    assert && assert( yAlign === 'top' || yAlign === 'center' || yAlign === 'bottom',
      'yAlign should be one of: \'top\', \'center\', or \'bottom\'' );

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
  private updateProperty( propName: 'left' | 'right' | 'top' | 'bottom' | 'centerX' | 'centerY', offset: number ) {
    const currentValue = this._content[ propName ];
    const newValue = this.localBounds[ propName ] + offset;

    // Prevent infinite loops or stack overflows by ignoring tiny changes
    if ( Math.abs( currentValue - newValue ) > 1e-5 ) {
      this._content[ propName ] = newValue;
    }
  }

  /**
   * Updates the layout of this alignment box.
   */
  private updateLayout() {
    if ( this._layoutLock ) { return; }
    this._layoutLock = true;

    sceneryLog && sceneryLog.AlignBox && sceneryLog.AlignBox( `AlignBox#${this.id} updateLayout` );
    sceneryLog && sceneryLog.AlignBox && sceneryLog.push();

    // If we have alignBounds, use that.
    if ( this._alignBounds !== null ) {
      this.localBounds = this._alignBounds;
    }
    // Otherwise, we'll grab a Bounds2 anchored at the upper-left with our required dimensions.
    else {
      const widthWithMargin = this._leftMargin + this._content.width + this._rightMargin;
      const heightWithMargin = this._topMargin + this._content.height + this._bottomMargin;
      this.localBounds = new Bounds2( 0, 0, widthWithMargin, heightWithMargin );
    }

    // Don't try to lay out empty bounds
    if ( !this._content.localBounds.isEmpty() ) {

      if ( this._xAlign === 'center' ) {
        this.updateProperty( 'centerX', ( this.leftMargin - this.rightMargin ) / 2 );
      }
      else if ( this._xAlign === 'left' ) {
        this.updateProperty( 'left', this._leftMargin );
      }
      else if ( this._xAlign === 'right' ) {
        this.updateProperty( 'right', -this._rightMargin );
      }
      else {
        assert && assert( `Bad xAlign: ${this._xAlign}` );
      }

      if ( this._yAlign === 'center' ) {
        this.updateProperty( 'centerY', ( this.topMargin - this.bottomMargin ) / 2 );
      }
      else if ( this._yAlign === 'top' ) {
        this.updateProperty( 'top', this._topMargin );
      }
      else if ( this._yAlign === 'bottom' ) {
        this.updateProperty( 'bottom', -this._bottomMargin );
      }
      else {
        assert && assert( `Bad yAlign: ${this._yAlign}` );
      }
    }

    sceneryLog && sceneryLog.AlignBox && sceneryLog.pop();

    this._layoutLock = false;
  }

  /**
   * Disposes this box, releasing listeners and any references to an AlignGroup
   */
  dispose() {
    // Remove our listener
    this._content.boundsProperty.unlink( this._contentBoundsListener );

    // Disconnects from the group
    this.group = null;

    super.dispose();
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @public
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
AlignBox.prototype._mutatorKeys = ALIGNMENT_CONTAINER_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'AlignBox', AlignBox );

export default AlignBox;
export type { AlignBoxOptions };
