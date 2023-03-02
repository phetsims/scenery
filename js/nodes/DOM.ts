// Copyright 2013-2023, University of Colorado Boulder

/**
 * Displays a DOM element directly in a node, so that it can be positioned/transformed properly, and bounds are handled properly in Scenery.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import { DOMDrawable, DOMSelfDrawable, Instance, Node, NodeOptions, Renderer, scenery } from '../imports.js';

const DOM_OPTION_KEYS = [
  'element', // {HTMLElement} - Sets the element, see setElement() for more documentation
  'preventTransform' // {boolean} - Sets whether Scenery is allowed to transform the element. see setPreventTransform() for docs
];

type SelfOptions = {
  element?: HTMLElement;
  preventTransform?: boolean;
};

export type DOMOptions = SelfOptions & NodeOptions;

// User-defined type guard
const isJQueryElement = ( element: Element | JQuery ): element is JQuery => !!( element && ( element as JQuery ).jquery );

export default class DOM extends Node {

  private _element!: HTMLElement;

  // Container div that will have our main element as a child (so we can position and mutate it).
  public readonly _container: HTMLDivElement; // scenery-internal

  // jQuery selection so that we can properly determine size information
  private readonly _$container: JQuery<HTMLDivElement>;

  // Flag that indicates whether we are updating/invalidating ourself due to changes to the DOM element. The flag is
  // needed so that updates to our element that we make in the update/invalidate section doesn't trigger an infinite
  // loop with another update.
  private invalidateDOMLock: boolean;

  // Flag that when true won't let Scenery apply a transform directly (the client will take care of that).
  private _preventTransform: boolean;

  /**
   * @param element - The HTML element, or a jQuery selector result.
   * @param [options] - DOM-specific options are documented in DOM_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  public constructor( element: Element | JQuery, options?: DOMOptions ) {
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    assert && assert( element instanceof window.Element || element.jquery, 'DOM nodes need to be passed an HTML/DOM element or a jQuery selection like $( ... )' ); // eslint-disable-line no-simple-type-checking-assertions

    // unwrap from jQuery if that is passed in, for consistency
    if ( isJQueryElement( element ) ) {
      element = element[ 0 ];

      assert && assert( element instanceof window.Element ); // eslint-disable-line no-simple-type-checking-assertions
    }

    super();

    this._container = document.createElement( 'div' );

    this._$container = $( this._container );
    this._$container.css( 'position', 'absolute' );
    this._$container.css( 'left', 0 );
    this._$container.css( 'top', 0 );

    this.invalidateDOMLock = false;
    this._preventTransform = false;

    // Have mutate() call setElement() in the proper order
    options = extendDefined( {
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
    const $element = $( this._element );
    return new Bounds2( 0, 0, $element.width()!, $element.height()! );
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

    // we will place ourselves in a temporary container to get our real desired bounds
    const temporaryContainer = document.createElement( 'div' );
    $( temporaryContainer ).css( {
      display: 'hidden',
      padding: '0 !important',
      margin: '0 !important',
      position: 'absolute',
      left: 0,
      top: 0,
      width: 65535,
      height: 65535
    } );

    // move to the temporary container
    this._container.removeChild( this._element );
    temporaryContainer.appendChild( this._element );
    document.body.appendChild( temporaryContainer );

    // bounds computation and resize our container to fit precisely
    const selfBounds = this.calculateDOMBounds();
    this.invalidateSelf( selfBounds );
    this._$container.width( selfBounds.getWidth() );
    this._$container.height( selfBounds.getHeight() );

    // move back to the main container
    document.body.removeChild( temporaryContainer );
    temporaryContainer.removeChild( this._element );
    this._container.appendChild( this._element );

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
