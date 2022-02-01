// Copyright 2016-2021, University of Colorado Boulder

/**
 * A group of alignment boxes that follow the constraints:
 * 1. Every box will have the same bounds, with an upper-left of (0,0)
 * 2. The box sizes will be the smallest possible to fit every box's content (with respective padding).
 * 3. Each box is responsible for positioning its content in its bounds (with customizable alignment and padding).
 *
 * Align boxes can be dynamically created and disposed, and only active boxes will be considered for the bounds.
 *
 * Since many sun components do not support resizing their contents dynamically, you may need to populate the AlignGroup
 * in the order of largest to smallest so that a fixed size container is large enough to contain the largest item.
 *
 * NOTE: Align box resizes may not happen immediately, and may be delayed until bounds of a align box's child occurs.
 *       layout updates can be forced with group.updateLayout(). If the align box's content that changed is connected
 *       to a Scenery display, its bounds will update when Display.updateDisplay() will called, so this will guarantee
 *       that the layout will be applied before it is displayed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IProperty from '../../../axon/js/IProperty.js';
import NumberProperty from '../../../axon/js/NumberProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import merge from '../../../phet-core/js/merge.js';
import optionize from '../../../phet-core/js/optionize.js';
import { scenery, Node, AlignBox } from '../imports.js';
import { AlignBoxOptions } from './AlignBox.js';

let globalId = 1;

type AlignGroupOptions = {
  // Whether the boxes should have all matching widths (otherwise it fits to size)
  matchHorizontal?: boolean,

  // Whether the boxes should have all matching heights (otherwise it fits to size)
  matchVertical?: boolean
};

const DEFAULT_OPTIONS = {
  matchHorizontal: true,
  matchVertical: true
};

class AlignGroup {

  private _alignBoxes: AlignBox[];
  private _matchHorizontal: boolean;
  private _matchVertical: boolean;

  // Gets locked when certain layout is performed.
  private _resizeLock: boolean;

  private _maxWidthProperty: NumberProperty;
  private _maxHeightProperty: NumberProperty;
  private id: number;

  /**
   * Creates an alignment group that can be composed of multiple boxes.
   *
   * Use createBox() to create alignment boxes. You can dispose() individual boxes, or call dispose() on this
   * group to dispose all of them.
   *
   * It is also possible to create AlignBox instances independently and assign their 'group' to this AlignGroup.
   */
  constructor( providedOptions?: AlignGroupOptions ) {
    assert && assert( providedOptions === undefined || Object.getPrototypeOf( providedOptions ) === Object.prototype,
      'Extra prototype on options object is a code smell' );

    const options = optionize<AlignGroupOptions, AlignGroupOptions>( {}, DEFAULT_OPTIONS, providedOptions );

    assert && assert( typeof options.matchHorizontal === 'boolean' );
    assert && assert( typeof options.matchVertical === 'boolean' );

    // @private {Array.<AlignBox>}
    this._alignBoxes = [];

    // @private {boolean}
    this._matchHorizontal = options.matchHorizontal;

    // @private {boolean}
    this._matchVertical = options.matchVertical;

    // @private {boolean} - Gets locked when certain layout is performed.
    this._resizeLock = false;

    // @private {Property.<boolean>}
    this._maxWidthProperty = new NumberProperty( 0 );

    // @private {Property.<boolean>}
    this._maxHeightProperty = new NumberProperty( 0 );

    // @private {number}
    this.id = globalId++;
  }

  /**
   * Returns the current maximum width of the grouped content.
   */
  getMaxWidth(): number {
    return this._maxWidthProperty.value;
  }

  get maxWidth(): number { return this.getMaxWidth(); }

  /**
   * Returns the Property holding the current maximum width of the grouped content.
   */
  getMaxWidthProperty(): IProperty<number> {
    return this._maxWidthProperty;
  }

  get maxWidthProperty(): IProperty<number> { return this.getMaxWidthProperty(); }

  /**
   * Returns the current maximum height of the grouped content.
   */
  getMaxHeight(): number {
    return this._maxHeightProperty.value;
  }

  get maxHeight(): number { return this.getMaxHeight(); }

  /**
   * Returns the Property holding the current maximum height of the grouped content.
   */
  getMaxHeightProperty(): IProperty<number> {
    return this._maxHeightProperty;
  }

  get maxHeightProperty(): IProperty<number> { return this.getMaxHeightProperty(); }

  /**
   * Creates an alignment box with the given content and options.
   */
  createBox( content: Node, options?: AlignBoxOptions ) {
    assert && assert( content instanceof Node );

    // Setting the group should call our addAlignBox()
    return new AlignBox( content, merge( {
      group: this
    }, options ) );
  }

  /**
   * Sets whether the widths of the align boxes should all match. If false, each box will use its preferred width
   * (usually equal to the content width + horizontal margins).
   */
  setMatchHorizontal( matchHorizontal: boolean ): this {
    assert && assert( typeof matchHorizontal === 'boolean' );

    if ( this._matchHorizontal !== matchHorizontal ) {
      this._matchHorizontal = matchHorizontal;

      // Update layout, since it will probably change
      this.updateLayout();
    }

    return this;
  }

  set matchHorizontal( value: boolean ) { this.setMatchHorizontal( value ); }

  /**
   * Returns whether boxes currently are horizontally matched. See setMatchHorizontal() for details.
   */
  getMatchHorizontal(): boolean {
    return this._matchHorizontal;
  }

  get matchHorizontal(): boolean { return this.getMatchHorizontal(); }

  /**
   * Sets whether the heights of the align boxes should all match. If false, each box will use its preferred height
   * (usually equal to the content height + vertical margins).
   */
  setMatchVertical( matchVertical: boolean ): this {
    assert && assert( typeof matchVertical === 'boolean' );

    if ( this._matchVertical !== matchVertical ) {
      this._matchVertical = matchVertical;

      // Update layout, since it will probably change
      this.updateLayout();
    }

    return this;
  }

  set matchVertical( value: boolean ) { this.setMatchVertical( value ); }

  /**
   * Returns whether boxes currently are vertically matched. See setMatchVertical() for details.
   */
  getMatchVertical(): boolean {
    return this._matchVertical;
  }

  get matchVertical(): boolean { return this.getMatchVertical(); }

  /**
   * Dispose all the boxes.
   */
  dispose() {
    for ( let i = this._alignBoxes.length - 1; i >= 0; i-- ) {
      this._alignBoxes[ i ].dispose();
    }
  }

  /**
   * Updates the localBounds and alignment for each alignBox.
   *
   * NOTE: Calling this will usually not be necessary outside of Scenery, but this WILL trigger bounds revalidation
   *       for every alignBox, which can force the layout code to run.
   */
  updateLayout() {
    if ( this._resizeLock ) { return; }
    this._resizeLock = true;

    sceneryLog && sceneryLog.AlignGroup && sceneryLog.AlignGroup(
      `AlignGroup#${this.id} updateLayout` );
    sceneryLog && sceneryLog.AlignGroup && sceneryLog.push();

    sceneryLog && sceneryLog.AlignGroup && sceneryLog.AlignGroup( 'AlignGroup computing maximum dimension' );
    sceneryLog && sceneryLog.AlignGroup && sceneryLog.push();

    // Compute the maximum dimension of our alignBoxs' content
    let maxWidth = 0;
    let maxHeight = 0;
    for ( let i = 0; i < this._alignBoxes.length; i++ ) {
      const alignBox = this._alignBoxes[ i ];

      const bounds = alignBox.getContentBounds();

      // Ignore bad bounds
      if ( bounds.isEmpty() || !bounds.isFinite() ) {
        continue;
      }

      maxWidth = Math.max( maxWidth, bounds.width );
      maxHeight = Math.max( maxHeight, bounds.height );
    }

    sceneryLog && sceneryLog.AlignGroup && sceneryLog.pop();
    sceneryLog && sceneryLog.AlignGroup && sceneryLog.AlignGroup( 'AlignGroup applying to boxes' );
    sceneryLog && sceneryLog.AlignGroup && sceneryLog.push();

    this._maxWidthProperty.value = maxWidth;
    this._maxHeightProperty.value = maxHeight;

    if ( maxWidth > 0 && maxHeight > 0 ) {
      // Apply that maximum dimension for each alignBox
      for ( let i = 0; i < this._alignBoxes.length; i++ ) {
        this.setBoxBounds( this._alignBoxes[ i ], maxWidth, maxHeight );
      }
    }

    sceneryLog && sceneryLog.AlignGroup && sceneryLog.pop();
    sceneryLog && sceneryLog.AlignGroup && sceneryLog.pop();

    this._resizeLock = false;
  }

  /**
   * Sets a box's bounds based on our maximum dimensions.
   */
  private setBoxBounds( alignBox: AlignBox, maxWidth: number, maxHeight: number ) {
    let alignBounds;

    // If we match both dimensions, we don't have to inspect the box's preferred size
    if ( this._matchVertical && this._matchHorizontal ) {
      alignBounds = new Bounds2( 0, 0, maxWidth, maxHeight );
    }
    else {
      // Grab the preferred size
      const contentBounds = alignBox.getContentBounds();

      // Match one orientation
      if ( this._matchVertical ) {
        alignBounds = new Bounds2( 0, 0, contentBounds.width, maxHeight );
      }
      else if ( this._matchHorizontal ) {
        alignBounds = new Bounds2( 0, 0, maxWidth, contentBounds.height );
      }
      // If not matching anything, just use its preferred size
      else {
        alignBounds = contentBounds;
      }
    }

    alignBox.alignBounds = alignBounds;
  }

  /**
   * Lets the group know that the alignBox has had its content resized. Called by the AlignBox (scenery-internal)
   */
  onAlignBoxResized( alignBox: AlignBox ) {
    // TODO: in the future, we could only update this specific alignBox if the others don't need updating.
    this.updateLayout();
  }

  /**
   * Adds the AlignBox to the group -- Used in AlignBox --- do NOT use in public code
   */
  addAlignBox( alignBox: AlignBox ) {
    this._alignBoxes.push( alignBox );

    // Trigger an update when a alignBox is added
    this.updateLayout();
  }

  /**
   * Removes the AlignBox from the group (scenery-internal)
   */
  removeAlignBox( alignBox: AlignBox ) {
    arrayRemove( this._alignBoxes, alignBox );

    // Trigger an update when a alignBox is removed
    this.updateLayout();
  }
}

scenery.register( 'AlignGroup', AlignGroup );
export default AlignGroup;
export type { AlignGroupOptions };
