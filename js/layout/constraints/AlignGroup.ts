// Copyright 2016-2023, University of Colorado Boulder

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

import TProperty from '../../../../axon/js/TProperty.js';
import NumberProperty from '../../../../axon/js/NumberProperty.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import arrayRemove from '../../../../phet-core/js/arrayRemove.js';
import { combineOptions, optionize3 } from '../../../../phet-core/js/optionize.js';
import { AlignBox, Node, scenery } from '../../imports.js';
import { AlignBoxOptions } from '../nodes/AlignBox.js';
import Disposable from '../../../../axon/js/Disposable.js';
import Orientation from '../../../../phet-core/js/Orientation.js';

let globalId = 1;

type SelfOptions = {
  // Whether the boxes should have all matching widths (otherwise it fits to size)
  matchHorizontal?: boolean;

  // Whether the boxes should have all matching heights (otherwise it fits to size)
  matchVertical?: boolean;
};

export type AlignGroupOptions = SelfOptions;

const DEFAULT_OPTIONS = {
  matchHorizontal: true,
  matchVertical: true
};

export default class AlignGroup extends Disposable {

  private readonly _alignBoxes: AlignBox[];
  private _matchHorizontal: boolean;
  private _matchVertical: boolean;

  // Gets locked when certain layout is performed.
  private _resizeLock: boolean;

  private readonly _maxWidthProperty: NumberProperty;
  private readonly _maxHeightProperty: NumberProperty;
  private readonly id: number;

  /**
   * Creates an alignment group that can be composed of multiple boxes.
   *
   * Use createBox() to create alignment boxes. You can dispose() individual boxes, or call dispose() on this
   * group to dispose all of them.
   *
   * It is also possible to create AlignBox instances independently and assign their 'group' to this AlignGroup.
   */
  public constructor( providedOptions?: AlignGroupOptions ) {
    assert && assert( providedOptions === undefined || Object.getPrototypeOf( providedOptions ) === Object.prototype,
      'Extra prototype on options object is a code smell' );

    const options = optionize3<AlignGroupOptions, SelfOptions>()( {}, DEFAULT_OPTIONS, providedOptions );

    assert && assert( typeof options.matchHorizontal === 'boolean' );
    assert && assert( typeof options.matchVertical === 'boolean' );

    super();

    this._alignBoxes = [];

    this._matchHorizontal = options.matchHorizontal;

    this._matchVertical = options.matchVertical;

    // Gets locked when certain layout is performed.
    this._resizeLock = false;

    this._maxWidthProperty = new NumberProperty( 0 );

    this._maxHeightProperty = new NumberProperty( 0 );

    this.id = globalId++;
  }

  /**
   * Returns the current maximum width of the grouped content.
   */
  public getMaxWidth(): number {
    return this._maxWidthProperty.value;
  }

  public get maxWidth(): number { return this.getMaxWidth(); }

  /**
   * Returns the Property holding the current maximum width of the grouped content.
   */
  public getMaxWidthProperty(): TProperty<number> {
    return this._maxWidthProperty;
  }

  public get maxWidthProperty(): TProperty<number> { return this.getMaxWidthProperty(); }

  /**
   * Returns the current maximum height of the grouped content.
   */
  public getMaxHeight(): number {
    return this._maxHeightProperty.value;
  }

  public get maxHeight(): number { return this.getMaxHeight(); }

  /**
   * Returns the Property holding the current maximum height of the grouped content.
   */
  public getMaxHeightProperty(): TProperty<number> {
    return this._maxHeightProperty;
  }

  public get maxHeightProperty(): TProperty<number> { return this.getMaxHeightProperty(); }

  public getMaxSizeProperty( orientation: Orientation ): TProperty<number> {
    return orientation === Orientation.HORIZONTAL ? this._maxWidthProperty : this._maxHeightProperty;
  }

  /**
   * Creates an alignment box with the given content and options.
   */
  public createBox( content: Node, options?: AlignBoxOptions ): AlignBox {

    // Setting the group should call our addAlignBox()
    return new AlignBox( content, combineOptions<AlignBoxOptions>( {
      group: this
    }, options ) );
  }

  /**
   * Sets whether the widths of the align boxes should all match. If false, each box will use its preferred width
   * (usually equal to the content width + horizontal margins).
   */
  public setMatchHorizontal( matchHorizontal: boolean ): this {

    if ( this._matchHorizontal !== matchHorizontal ) {
      this._matchHorizontal = matchHorizontal;

      // Update layout, since it will probably change
      this.updateLayout();
    }

    return this;
  }

  public set matchHorizontal( value: boolean ) { this.setMatchHorizontal( value ); }

  public get matchHorizontal(): boolean { return this.getMatchHorizontal(); }

  /**
   * Returns whether boxes currently are horizontally matched. See setMatchHorizontal() for details.
   */
  public getMatchHorizontal(): boolean {
    return this._matchHorizontal;
  }

  /**
   * Sets whether the heights of the align boxes should all match. If false, each box will use its preferred height
   * (usually equal to the content height + vertical margins).
   */
  public setMatchVertical( matchVertical: boolean ): this {

    if ( this._matchVertical !== matchVertical ) {
      this._matchVertical = matchVertical;

      // Update layout, since it will probably change
      this.updateLayout();
    }

    return this;
  }

  public set matchVertical( value: boolean ) { this.setMatchVertical( value ); }

  public get matchVertical(): boolean { return this.getMatchVertical(); }

  /**
   * Returns whether boxes currently are vertically matched. See setMatchVertical() for details.
   */
  public getMatchVertical(): boolean {
    return this._matchVertical;
  }

  /**
   * Updates the localBounds and alignment for each alignBox.
   *
   * NOTE: Calling this will usually not be necessary outside of Scenery, but this WILL trigger bounds revalidation
   *       for every alignBox, which can force the layout code to run.
   */
  public updateLayout(): void {
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
  private setBoxBounds( alignBox: AlignBox, maxWidth: number, maxHeight: number ): void {
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
        alignBounds = new Bounds2( 0, 0, isFinite( contentBounds.width ) ? contentBounds.width : maxWidth, maxHeight );
      }
      else if ( this._matchHorizontal ) {
        alignBounds = new Bounds2( 0, 0, maxWidth, isFinite( contentBounds.height ) ? contentBounds.height : maxHeight );
      }
      // If not matching anything, just use its preferred size
      else {
        alignBounds = contentBounds;
      }
    }

    alignBox.alignBounds = alignBounds;
  }

  /**
   * Lets the group know that the alignBox has had its content resized. Called by the AlignBox
   * (scenery-internal)
   */
  public onAlignBoxResized( alignBox: AlignBox ): void {
    // NOTE: in the future, we could only update this specific alignBox if the others don't need updating.
    this.updateLayout();
  }

  /**
   * Adds the AlignBox to the group -- Used in AlignBox --- do NOT use in public code
   * (scenery-internal)
   */
  public addAlignBox( alignBox: AlignBox ): void {
    this._alignBoxes.push( alignBox );

    // Trigger an update when a alignBox is added
    this.updateLayout();
  }

  /**
   * Removes the AlignBox from the group
   * (scenery-internal)
   */
  public removeAlignBox( alignBox: AlignBox ): void {
    arrayRemove( this._alignBoxes, alignBox );

    // Trigger an update when a alignBox is removed
    this.updateLayout();
  }

  /**
   * Dispose all the boxes.
   */
  public override dispose(): void {
    for ( let i = this._alignBoxes.length - 1; i >= 0; i-- ) {
      this._alignBoxes[ i ].dispose();
    }
    super.dispose();
  }
}

scenery.register( 'AlignGroup', AlignGroup );
