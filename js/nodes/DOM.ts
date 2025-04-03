// Copyright 2013-2025, University of Colorado Boulder

/**
 * Displays a DOM element directly in a node, so that it can be positioned/transformed properly, and bounds are handled properly in Scenery.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import DOMDrawable from '../display/drawables/DOMDrawable.js';
import DOMSelfDrawable from '../display/DOMSelfDrawable.js';
import Instance from '../display/Instance.js';
import type { NodeOptions } from '../nodes/Node.js';
import Node from '../nodes/Node.js';
import Renderer from '../display/Renderer.js';
import scenery from '../scenery.js';

const DOM_OPTION_KEYS = [
  'element',
  'preventTransform',
  'allowInput'
];

type SelfOptions = {
  // Sets the element, see setElement() for more documentation
  element?: Element;

  // Sets whether Scenery is allowed to transform the element. see setPreventTransform() for docs
  preventTransform?: boolean;

  // Whether we allow input to be received by the DOM element
  allowInput?: boolean;
};

export type DOMOptions = SelfOptions & NodeOptions;

let boundsMeasurementContainer: HTMLDivElement | null = null;

export default class DOM extends Node {

  private _element!: HTMLElement;

  // Container div that will have our main element as a child (so we can position and mutate it).
  public readonly _container: HTMLDivElement; // scenery-internal

  // Flag that indicates whether we are updating/invalidating ourself due to changes to the DOM element. The flag is
  // needed so that updates to our element that we make in the update/invalidate section doesn't trigger an infinite
  // loop with another update.
  private invalidateDOMLock: boolean;

  // Flag that when true won't let Scenery apply a transform directly (the client will take care of that).
  private _preventTransform: boolean;

  // Whether we allow input to be received by the DOM element
  private _allowInput: boolean;

  /**
   * @param element - The HTML element, or a jQuery selector result.
   * @param [options] - DOM-specific options are documented in DOM_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  public constructor( element: Element, options?: DOMOptions ) {
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    super();

    this._container = document.createElement( 'div' );
    this._container.style.position = 'absolute';
    this._container.style.left = '0';
    this._container.style.top = '0';
    this._container.style.setProperty( 'padding', '0', 'important' );
    this._container.style.setProperty( 'margin', '0', 'important' );
    this._allowInput = false;

    this.invalidateDOMLock = false;
    this._preventTransform = false;

    this.invalidateAllowInput( this._allowInput );

    // Have mutate() call setElement() in the proper order
    options = extendDefined<DOMOptions>( {
      element: element
    }, options );

    // will set the element after initializing
    this.mutate( options );

    // Only renderer supported, no need to dynamically compute
    this.setRendererBitmask( Renderer.bitmaskDOM );
  }


  /**
   * Computes the bounds of our current DOM element (using jQuery, as replacing this with other things seems a bit
   * bug-prone and has caused issues in the past).
   *
   * The dom element needs to be attached to the DOM tree in order for this to work.
   *
   * Alternative getBoundingClientRect explored, but did not seem sufficient (possibly due to CSS transforms)?
   */
  protected calculateDOMBounds(): Bounds2 {
    return new Bounds2( 0, 0, this._element.offsetWidth, this._element.offsetHeight );
  }

  /**
   * Triggers recomputation of our DOM element's bounds.
   *
   * This should be called after the DOM element's bounds may have changed, to properly update the bounding box
   * in Scenery.
   */
  public invalidateDOM(): void {
    // prevent this from being executed as a side-effect from inside one of its own calls
    if ( this.invalidateDOMLock ) {
      return;
    }
    this.invalidateDOMLock = true;

    const needsReparenting = !this._container.isConnected;

    if ( needsReparenting ) {
      if ( !boundsMeasurementContainer ) {
        boundsMeasurementContainer = document.createElement( 'div' );
        boundsMeasurementContainer.style.visibility = 'hidden';
        boundsMeasurementContainer.style.position = 'absolute'; // using offset dimensions, so we need this to be positioned
        boundsMeasurementContainer.style.left = '0px';
        boundsMeasurementContainer.style.top = '0px';
        boundsMeasurementContainer.style.setProperty( 'padding', '0', 'important' );
        boundsMeasurementContainer.style.setProperty( 'margin', '0', 'important' );
      }

      // move to the temporary container
      this._container.removeChild( this._element );
      boundsMeasurementContainer.appendChild( this._element );
      document.body.appendChild( boundsMeasurementContainer );
    }

    // bounds computation and resize our container to fit precisely
    const selfBounds = this.calculateDOMBounds();
    this.invalidateSelf( selfBounds );

    if ( needsReparenting ) {
      document.body.removeChild( boundsMeasurementContainer! );
      boundsMeasurementContainer!.removeChild( this._element );
      this._container.appendChild( this._element );
    }

    // unlock
    this.invalidateDOMLock = false;
  }

  /**
   * Creates a DOM drawable for this DOM node. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createDOMDrawable( renderer: number, instance: Instance ): DOMSelfDrawable {
    // @ts-expect-error Poolable
    return DOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Whether this Node itself is painted (displays something itself).
   */
  public override isPainted(): boolean {
    // Always true for DOM nodes
    return true;
  }

  /**
   * Changes the DOM element of this DOM node to another element.
   */
  public setElement( element: HTMLElement ): this {
    assert && assert( !this._element, 'We should only ever attach one DOMElement to a DOM node' );

    if ( this._element !== element ) {
      if ( this._element ) {
        this._container.removeChild( this._element );
      }

      this._element = element;

      this._container.appendChild( this._element );

      this.invalidateDOM();
    }

    return this; // allow chaining
  }

  public set element( value: HTMLElement ) { this.setElement( value ); }

  public get element(): HTMLElement { return this.getElement(); }

  /**
   * Returns the DOM element being displayed by this DOM node.
   */
  public getElement(): HTMLElement {
    return this._element;
  }

  /**
   * Sets the value of the preventTransform flag.
   *
   * When the preventTransform flag is set to true, Scenery will not reposition (CSS transform) the DOM element, but
   * instead it will be at the upper-left (0,0) of the Scenery Display. The client will be responsible for sizing or
   * positioning this element instead.
   */
  public setPreventTransform( preventTransform: boolean ): void {
    if ( this._preventTransform !== preventTransform ) {
      this._preventTransform = preventTransform;
    }
  }

  public set preventTransform( value: boolean ) { this.setPreventTransform( value ); }

  public get preventTransform(): boolean { return this.isTransformPrevented(); }

  /**
   * Returns the value of the preventTransform flag.
   *
   * See the setPreventTransform documentation for more information on the flag.
   */
  public isTransformPrevented(): boolean {
    return this._preventTransform;
  }

  /**
   * Sets whether input is allowed for the DOM element. If false, we will disable input events with pointerEvents and
   * the usual preventDefault(). If true, we'll set a flag internally so that we don't preventDefault() on input events.
   */
  public setAllowInput( allowInput: boolean ): void {
    if ( this._allowInput !== allowInput ) {
      this._allowInput = allowInput;

      this.invalidateAllowInput( allowInput );
    }
  }

  public isInputAllowed(): boolean { return this._allowInput; }

  private invalidateAllowInput( allowInput: boolean ): void {
    this._container.dataset.sceneryAllowInput = allowInput ? 'true' : 'false';
    this._container.style.pointerEvents = allowInput ? 'auto' : 'none';
  }

  public set allowInput( value: boolean ) { this.setAllowInput( value ); }

  public get allowInput(): boolean { return this.isInputAllowed(); }

  public override mutate( options?: DOMOptions ): this {
    return super.mutate( options );
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
DOM.prototype._mutatorKeys = DOM_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'DOM', DOM );