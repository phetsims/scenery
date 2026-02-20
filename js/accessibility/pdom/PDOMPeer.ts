// Copyright 2015-2026, University of Colorado Boulder

/**
 * An accessible peer controls the appearance of an accessible Node's instance in the parallel DOM. A PDOMPeer can
 * have up to four window.Elements displayed in the PDOM, see constructor for details.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 * @author Jesse Greenberg
 */

import arrayRemove from '../../../../phet-core/js/arrayRemove.js';
import merge from '../../../../phet-core/js/merge.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import stripEmbeddingMarks from '../../../../phet-core/js/stripEmbeddingMarks.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import FocusManager from '../../accessibility/FocusManager.js';
import PDOMSiblingStyle from '../../accessibility/pdom/PDOMSiblingStyle.js';
import PDOMUtils from '../../accessibility/pdom/PDOMUtils.js';
import type Display from '../../display/Display.js';
import type Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import type Trail from '../../util/Trail.js';
import { pdomFocusProperty } from '../pdomFocusProperty.js';
import { guessVisualTrail } from './guessVisualTrail.js';
import type PDOMInstance from './PDOMInstance.js';
import { PEER_ACCESSIBLE_PARAGRAPH_SIBLING } from './PEER_ACCESSIBLE_PARAGRAPH_SIBLING.js';
import { PEER_CONTAINER_PARENT } from './PEER_CONTAINER_PARENT.js';
import { PEER_DESCRIPTION_SIBLING } from './PEER_DESCRIPTION_SIBLING.js';
import { PEER_HEADING_SIBLING } from './PEER_HEADING_SIBLING.js';
import { PEER_LABEL_SIBLING } from './PEER_LABEL_SIBLING.js';
import { PEER_PRIMARY_SIBLING } from './PEER_PRIMARY_SIBLING.js';

// constants
const PRIMARY_SIBLING = PEER_PRIMARY_SIBLING;
const HEADING_SIBLING = PEER_HEADING_SIBLING;
const LABEL_SIBLING = PEER_LABEL_SIBLING;
const DESCRIPTION_SIBLING = PEER_DESCRIPTION_SIBLING;
const ACCESSIBLE_PARAGRAPH_SIBLING = PEER_ACCESSIBLE_PARAGRAPH_SIBLING;
const CONTAINER_PARENT = PEER_CONTAINER_PARENT;
const LABEL_TAG = PDOMUtils.TAGS.LABEL;
const INPUT_TAG = PDOMUtils.TAGS.INPUT;
const DISABLED_ATTRIBUTE_NAME = 'disabled';

let globalId = 1;

/**
 * @mixes Poolable
 */
class PDOMPeer {

  // unique ID
  public id!: number;

  public pdomInstance!: PDOMInstance;

  // only null for the root pdomInstance
  public node!: Node | null;

  // Each peer is associated with a specific Display.
  public display!: Display;

  // NOTE: May have "gaps" due to pdomOrder usage.
  public trail!: Trail;

  // whether this PDOMPeer is visible in the PDOM
  // Only initialized to null, should not be set to it. isVisible() will return true if this.visible is null
  // (because it hasn't been set yet).
  public visible!: boolean | null;

  // whether the primary sibling can receive focus.
  public focusable!: boolean | null;

  // Optional label/description elements
  private _headingSibling!: HTMLElement | null;
  private _labelSibling!: HTMLElement | null;
  private _descriptionSibling!: HTMLElement | null;

  // Optional paragraph element for the "high level" API.
  private _accessibleParagraphSibling!: HTMLElement | null;

  // A parent element that can contain this primarySibling and other siblings, usually
  // the label and description content.
  private _containerParent!: HTMLElement | null;

  // Rather than guarantee that a peer is a tree with a root DOMElement,
  // allow multiple window.Elements at the top level of the peer. This is used for sorting the instance.
  // See this.orderElements for more info.
  public topLevelElements!: HTMLElement[];

  // To support setting the Display.interactive=false (which sets disabled on all primarySiblings,
  // we need to set disabled on a separate channel from this.setAttributeToElement. That way we cover the case where
  // `disabled` was set through the ParallelDOM API when we need to toggle it specifically for Display.interactive.
  // This way we can conserve the previous `disabled` attribute/property value through toggling Display.interactive.
  private _preservedDisabledValue: IntentionalAny | null;

  // Whether we are currently in a "disposed" (in the pool) state, or are available to be interacted with.
  private isDisposed!: boolean;

  // The main element associated with this peer. If focusable, this is the element that gets
  // the focus. It also will contain any children.
  private _primarySibling!: HTMLElement | undefined;

  protected constructor( pdomInstance: PDOMInstance, options: IntentionalAny ) {
    this.initializePDOMPeer( pdomInstance, options );
  }


  /**
   * Initializes the object (either from a freshly-created state, or from a "disposed" state brought back from a
   * pool).
   *
   * NOTE: the PDOMPeer is not fully constructed until calling PDOMPeer.update() after creating from pool.
   *
   * @returns 'this' reference, for chaining
   * (scenery-internal)
   */
  public initializePDOMPeer( pdomInstance: PDOMInstance, options: IntentionalAny ): this {

    // eslint-disable-next-line phet/bad-typescript-text
    options = merge( {
      primarySibling: null
    }, options );

    assert && assert( !this.id || this.isDisposed, 'If we previously existed, we need to have been disposed' );

    this.id = this.id || globalId++;
    this.pdomInstance = pdomInstance;
    this.node = this.pdomInstance.node;
    this.display = pdomInstance.display!;
    this.trail = pdomInstance.trail!;
    this.visible = null;
    this.focusable = null;

    this._headingSibling = null;
    this._labelSibling = null;
    this._descriptionSibling = null;

    this._accessibleParagraphSibling = null;
    this._containerParent = null;
    this.topLevelElements = [];

    this._preservedDisabledValue = null;

    this.isDisposed = false;

    // edge case for root accessibility
    if ( this.pdomInstance.isRootInstance ) {

      this._primarySibling = options.primarySibling;
      this._primarySibling!.classList.add( PDOMSiblingStyle.ROOT_CLASS_NAME );

      // Stop blocked events from bubbling past the root of the PDOM so that scenery does
      // not dispatch them in Input.js.
      PDOMUtils.BLOCKED_DOM_EVENTS.forEach( eventType => {
        this._primarySibling!.addEventListener( eventType, event => {
          event.stopPropagation();
        } );
      } );
    }

    return this;
  }

  /**
   * Update the content of the peer. This must be called after the AccessiblePeer is constructed from pool.
   * @param updateIndicesStringAndElementIds - if this function should be called upon initial "construction" (in update), allows for the option to do this lazily, see https://github.com/phetsims/phet-io/issues/1847
   * (scenery-internal)
   */
  public update( updateIndicesStringAndElementIds: boolean ): void {

    // NOTE: JO: 2025-04-01: This looks like it will usually convert string Properties to strings (because of the getters)
    // but this will probably make TypeScript conversion problematic.
    let options = this.node!.getBaseOptions();

    const callbacksForOtherNodes: Array<() => void> = [];

    // Even if the accessibleName is null, we need to run the behavior function if the dirty flag is set
    // to run any cleanup on Nodes changed with callbacksForOtherNodes. See https://github.com/phetsims/scenery/issues/1679.
    if ( this.node!.accessibleName !== null || this.node!._accessibleNameDirty ) {
      if ( this.node!.accessibleName === null ) {

        // There is no accessibleName, so we don't want to modify options - just run the behavior for cleanup
        // and to update other nodes.
        this.node!.accessibleNameBehavior( this.node!, {}, this.node!.accessibleName, callbacksForOtherNodes );
      }
      else {
        options = this.node!.accessibleNameBehavior( this.node!, options, this.node!.accessibleName, callbacksForOtherNodes );
      }
      // eslint-disable-next-line phet/no-simple-type-checking-assertions
      assert && assert( typeof options === 'object', 'should return an object' );
      this.node!._accessibleNameDirty = false;
    }

    // Even if the accessibleHelpText is null, we need to run the behavior function if the dirty flag is set
    // to run any cleanup on Nodes changed with callbacksForOtherNodes. See https://github.com/phetsims/scenery/issues/1679.
    if ( this.node!.accessibleHelpText !== null || this.node!._accessibleHelpTextDirty ) {
      if ( this.node!.accessibleHelpText === null ) {

        // There is no accessibleHelpText, so we don't want to modify options - just run the behavior for cleanup
        // and to update other nodes.
        this.node!.accessibleHelpTextBehavior( this.node!, {}, this.node!.accessibleHelpText, callbacksForOtherNodes );
      }
      else {
        options = this.node!.accessibleHelpTextBehavior( this.node!, options, this.node!.accessibleHelpText, callbacksForOtherNodes );
      }
      // eslint-disable-next-line phet/no-simple-type-checking-assertions
      assert && assert( typeof options === 'object', 'should return an object' );
      this.node!._accessibleHelpTextDirty = false;
    }

    // Even if the accessibleHelpText is null, we need to run the behavior function if the dirty flag is set
    // to run any cleanup on Nodes changed with callbacksForOtherNodes. See https://github.com/phetsims/scenery/issues/1679.
    if ( this.node!.accessibleParagraph !== null || this.node!._accessibleParagraphDirty ) {
      if ( this.node!.accessibleParagraph === null ) {

        // There is no accessibleParagraph, so we don't want to modify options - just run the behavior for cleanup
        // and to update other nodes.
        this.node!.accessibleParagraphBehavior( this.node!, {}, this.node!.accessibleParagraph, callbacksForOtherNodes );
      }
      else {
        options = this.node!.accessibleParagraphBehavior( this.node!, options, this.node!.accessibleParagraph, callbacksForOtherNodes );
      }
      // eslint-disable-next-line phet/no-simple-type-checking-assertions
      assert && assert( typeof options === 'object', 'should return an object' );
      this.node!._accessibleParagraphDirty = false;
    }

    // If there is an accessible paragraph, create it now. Accessible paragraph does not require a tagName.
    //
    // Always create a paragraph element, even if accessibleParagraphContent is missing.
    // This ensures a top-level element exists for the peer, which is needed for cases
    // like "forwarding" to other Nodes via callbacksForOtherNodes. In such cases, the
    // paragraph will be empty because this Node has no content of its own.
    // If the content is an empty string, we still need to create an element for it, otherwise the element won't
    // be available when the content is populated without calling update() again.
    const paragraphContentDefined = options.accessibleParagraphContent !== null && options.accessibleParagraphContent !== undefined;
    const paragraphDefined = options.accessibleParagraph !== null && options.accessibleParagraph !== undefined;
    if ( paragraphContentDefined || paragraphDefined ) {
      this._accessibleParagraphSibling = createElement( PDOMUtils.TAGS.P, false );

      // @ts-expect-error
      this.setAccessibleParagraphContent( options.accessibleParagraphContent );
    }

    // accessibleHeading and accessibleParagraph can be used without a tagName, and that it enables PDOM for the Node.
    // If there is no tagName, create one so that children are placed under it by default.
    // This is done in PDOMPeer instead of on the ParallelDOM, so that we do not alter public state.
    // NOTE: It is posssible that the children check is enough. But there are assumptions in this file that
    // a component using accessibleHeading always has a tagName for children (like getPlaceableSibling).
    const headingRequiresContainer = options.accessibleHeading && !options.tagName;
    const paragraphRequiresContainer = this.pdomInstance.children.length > 0 && !options.tagName;
    if ( headingRequiresContainer || paragraphRequiresContainer ) {
      options.tagName = PDOMUtils.TAGS.DIV;
    }

    // create the base DOM element representing this accessible instance
    if ( options.tagName ) {

      // TODO: why not just options.focusable? https://github.com/phetsims/scenery/issues/1581
      this._primarySibling = createElement( options.tagName, this.node!.focusable, {
        namespace: options.pdomNamespace
      } );

      // create the container parent for the dom siblings
      if ( options.containerTagName ) {
        this._containerParent = createElement( options.containerTagName, false );
      }

      if ( options.accessibleHeading ) {
        this._headingSibling = createElement( `h${this.pdomInstance.pendingHeadingLevel}`, false );
      }

      // create the label DOM element representing this instance
      if ( options.labelTagName ) {
        this._labelSibling = createElement( options.labelTagName, false, {
          excludeFromInput: this.node!._excludeLabelSiblingFromInput
        } );
      }

      // create the description DOM element representing this instance
      if ( options.descriptionTagName ) {
        this._descriptionSibling = createElement( options.descriptionTagName, false );
      }
    }

    // Set ids on the elements - Attributes set below this call may require that the ids are set!!
    updateIndicesStringAndElementIds && this.updateIndicesStringAndElementIds();

    this.orderElements( options );

    // New siblings must remain hidden if the peer already was.
    if ( this.visible === false ) {
      for ( let i = 0; i < this.topLevelElements.length; i++ ) {
        this.setAttributeToElement( 'hidden', '', { element: this.topLevelElements[ i ] } );
      }
    }

    // The primary sibling (set with Node.tagName) is required for the peer to be visible in the PDOM.
    if ( this._primarySibling ) {

      // If Display.interactive is off, restore the disabled state on the new primary sibling.
      if ( !this.display.interactive ) {
        // @ts-expect-error
        this._primarySibling.disabled = true;
      }

      if ( options.accessibleHeading !== null ) {
        // @ts-expect-error
        this.setHeadingContent( options.accessibleHeading );
      }

      // set the accessible label now that the element has been recreated again, but not if the tagName
      // has been cleared out
      if ( options.labelContent && options.labelTagName !== null ) {
        // @ts-expect-error
        this.setLabelSiblingContent( options.labelContent );
      }

      // restore the innerContent
      if ( options.innerContent && options.tagName !== null ) {
        // @ts-expect-error
        this.setPrimarySiblingContent( options.innerContent );
      }

      // set the accessible description, but not if the tagName has been cleared out.
      if ( options.descriptionContent && options.descriptionTagName !== null ) {
        // @ts-expect-error
        this.setDescriptionSiblingContent( options.descriptionContent );
      }

      // if element is an input element, set input type
      if ( options.tagName!.toUpperCase() === INPUT_TAG && options.inputType ) {
        this.setAttributeToElement( 'type', options.inputType );
      }

      // if the label element happens to be a 'label', associate with 'for' attribute (must be done after updating IDs)
      if ( options.labelTagName && options.labelTagName.toUpperCase() === LABEL_TAG ) {
        this.setAttributeToElement( 'for', this._primarySibling.id, {
          elementName: PDOMPeer.LABEL_SIBLING
        } );
      }

      this.setFocusable( this.node!.focusable );

      // recompute and assign the association attributes that link two elements (like aria-labelledby)
      this.onAriaLabelledbyAssociationChange();
      this.onAriaDescribedbyAssociationChange();
      this.onActiveDescendantAssociationChange();

      // update all attributes for the peer, should cover aria-label, role, and others
      this.onAttributeChange( options );

      // update all classes for the peer
      this.onClassChange();

      // update input value attribute for the peer
      this.onInputValueChange();

      this.node!.updateOtherNodesAriaLabelledby();
      this.node!.updateOtherNodesAriaDescribedby();
      this.node!.updateOtherNodesActiveDescendant();
    }

    callbacksForOtherNodes.forEach( callback => callback() );
  }

  /**
   * Handle the internal ordering of the elements in the peer, this involves setting the proper value of
   * this.topLevelElements
   * @param config - the computed mixin options to be applied to the peer. (select ParallelDOM mutator keys)
   */
  private orderElements( config: IntentionalAny ): void {
    if ( this._containerParent ) {
      // The first child of the container parent element should be the peer dom element
      // if undefined, the insertBefore method will insert the this._primarySibling as the first child
      assert && assert( this._primarySibling, 'There should be a _primarySibling if there is a _containerParent' );
      this._containerParent.insertBefore( this._primarySibling!, this._containerParent.children[ 0 ] || null );
      this.topLevelElements = [ this._containerParent ];
    }
    else {

      // Wean out any null siblings
      this.topLevelElements = [ this._headingSibling, this._labelSibling, this._descriptionSibling, this._primarySibling, this._accessibleParagraphSibling ].filter( _.identity ) as HTMLElement[];
    }

    // insert the heading/label/description elements in the correct location if they exist
    // NOTE: Important for arrangeContentElement to be called in this order. This should ensure that the
    // heading sibling is first (if present)
    this._headingSibling && this.arrangeContentElement( this._headingSibling, false );
    if ( this._accessibleParagraphSibling ) {
      let appendAccessibleParagraph = config.appendAccessibleParagraph;

      // Without a primary sibling, prepend would assert. Always append in that case.
      if ( !this._primarySibling ) {
        appendAccessibleParagraph = true;
      }

      this.arrangeContentElement( this._accessibleParagraphSibling, appendAccessibleParagraph );
    }
    this._labelSibling && this.arrangeContentElement( this._labelSibling, config.appendLabel );
    this._descriptionSibling && this.arrangeContentElement( this._descriptionSibling, config.appendDescription );
  }

  /**
   * Returns the sibling that can be placed in the PDOM. This is the primary sibling if it exists, otherwise the
   * accessible paragraph.
   *
   * If other elements can be placed without the primary sibling, they could be added to this function.
   */
  public getPlaceableSibling(): IntentionalAny {
    const placeable = this._primarySibling || this._accessibleParagraphSibling;
    assert && assert( placeable, 'No placeable sibling found!' );

    return placeable;
  }

  /**
   * Get the primary sibling HTMLElement for the peer
   */
  public getPrimarySibling(): HTMLElement | null {
    return this._primarySibling || null;
  }

  public get primarySibling(): HTMLElement | null { return this.getPrimarySibling(); }

  /**
   * Get the heading sibling element for the peer
   */
  public getHeadingSibling(): HTMLElement | null {
    return this._headingSibling || null;
  }

  public get headingSibling(): HTMLElement | null { return this.getHeadingSibling(); }

  /**
   * Get the label sibling element for the peer
   */
  public getLabelSibling(): HTMLElement | null {
    return this._labelSibling || null;
  }

  public get labelSibling(): HTMLElement | null { return this.getLabelSibling(); }

  /**
   * Get the description sibling element for the peer
   */
  public getDescriptionSibling(): HTMLElement | null {
    return this._descriptionSibling || null;
  }

  public get descriptionSibling(): HTMLElement | null { return this.getDescriptionSibling(); }

  /**
   * Get the container parent element for the peer
   */
  public getContainerParent(): HTMLElement | null {
    return this._containerParent || null;
  }

  public get containerParent(): HTMLElement | null { return this.getContainerParent(); }

  /**
   * Recompute the aria-labelledby attributes for all of the peer's elements
   */
  public onAriaLabelledbyAssociationChange(): void {
    this.removeAttributeFromAllElements( 'aria-labelledby' );

    for ( let i = 0; i < this.node!.ariaLabelledbyAssociations.length; i++ ) {
      const associationObject = this.node!.ariaLabelledbyAssociations[ i ];

      // Assert out if the model list is different than the data held in the associationObject
      assert && assert( associationObject.otherNode.nodesThatAreAriaLabelledbyThisNode.includes( this.node! ),
        'unexpected otherNode' );


      this.setAssociationAttribute( 'aria-labelledby', associationObject );
    }
  }

  /**
   * Recompute the aria-describedby attributes for all of the peer's elements
   */
  public onAriaDescribedbyAssociationChange(): void {
    this.removeAttributeFromAllElements( 'aria-describedby' );

    for ( let i = 0; i < this.node!.ariaDescribedbyAssociations.length; i++ ) {
      const associationObject = this.node!.ariaDescribedbyAssociations[ i ];

      // Assert out if the model list is different than the data held in the associationObject
      assert && assert( associationObject.otherNode.nodesThatAreAriaDescribedbyThisNode.includes( this.node! ),
        'unexpected otherNode' );


      this.setAssociationAttribute( 'aria-describedby', associationObject );
    }
  }

  /**
   * Recompute the aria-activedescendant attributes for all of the peer's elements
   */
  public onActiveDescendantAssociationChange(): void {
    this.removeAttributeFromAllElements( 'aria-activedescendant' );

    for ( let i = 0; i < this.node!.activeDescendantAssociations.length; i++ ) {
      const associationObject = this.node!.activeDescendantAssociations[ i ];

      // Assert out if the model list is different from the data held in the associationObject
      assert && assert( associationObject.otherNode.nodesThatAreActiveDescendantToThisNode.includes( this.node! ),
        'unexpected otherNode' );


      this.setAssociationAttribute( 'aria-activedescendant', associationObject );
    }
  }

  /**
   * Set the new attribute to the element if the value is a string. It will otherwise be null or undefined and should
   * then be removed from the element. This allows empty strings to be set as values.
   */
  private handleAttributeWithPDOMOption( key: string, value: string | null | undefined ): void {
    if ( typeof value === 'string' ) {
      this.setAttributeToElement( key, value );
    }
    else {
      this.removeAttributeFromElement( key );
    }
  }

  /**
   * Set all pdom attributes onto the peer elements from the model's stored data objects
   *
   * @param [pdomOptions] - these can override the values of the node, see this.update()
   */
  private onAttributeChange( pdomOptions?: IntentionalAny ): void {

    for ( let i = 0; i < this.node!.pdomAttributes.length; i++ ) {
      const dataObject = this.node!.pdomAttributes[ i ];
      this.setAttributeToElement( dataObject.attribute, dataObject.value, dataObject.options );
    }

    // Manually support options that map to attributes. This covers that case where behavior functions want to change
    // these, but they aren't in node.pdomAttributes. It will do double work in some cases, but it is pretty minor for
    // the complexity it saves. https://github.com/phetsims/scenery/issues/1436. Empty strings should be settable for
    // these attributes but null and undefined are ignored.
    this.handleAttributeWithPDOMOption( 'aria-label', pdomOptions.ariaLabel );
    this.handleAttributeWithPDOMOption( 'role', pdomOptions.ariaRole );
  }

  /**
   * Set all classes onto the peer elements from the model's stored data objects
   */
  private onClassChange(): void {
    for ( let i = 0; i < this.node!.pdomClasses.length; i++ ) {
      const dataObject = this.node!.pdomClasses[ i ];
      this.setClassToElement( dataObject.className, dataObject.options );
    }
  }

  /**
   * Set the input value on the peer's primary sibling HTMLElement. The value attribute must be set as a Property to be
   * registered correctly by an assistive device. If null, the attribute is removed so that we don't clutter the DOM
   * with value="null" attributes.
   *
   * (scenery-internal)
   */
  public onInputValueChange(): void {
    assert && assert( this.node!.inputValue !== undefined, 'use null to remove input value attribute' );

    if ( this.node!.inputValue === null ) {
      this.removeAttributeFromElement( 'value' );
    }
    else {

      // type conversion for DOM spec
      const valueString = `${this.node!.inputValue}`;
      this.setAttributeToElement( 'value', valueString, { type: 'property' } );
    }
  }

  /**
   * Get an element on this node, looked up by the elementName flag passed in.
   * (scenery-internal)
   *
   * @param elementName - see PDOMUtils for valid associations
   */
  public getElementByName( elementName: string ): HTMLElement {
    if ( elementName === PDOMPeer.PRIMARY_SIBLING ) {
      return this._primarySibling!;
    }
    else if ( elementName === PDOMPeer.LABEL_SIBLING ) {
      return this._labelSibling!;
    }
    else if ( elementName === PDOMPeer.DESCRIPTION_SIBLING ) {
      return this._descriptionSibling!;
    }
    else if ( elementName === PDOMPeer.CONTAINER_PARENT ) {
      return this._containerParent!;
    }
    else if ( elementName === ACCESSIBLE_PARAGRAPH_SIBLING ) {
      return this._accessibleParagraphSibling!;
    }
    else if ( elementName === HEADING_SIBLING ) {
      return this._headingSibling!;
    }

    throw new Error( `invalid elementName name: ${elementName}` );
  }

  /**
   * Sets a attribute on one of the peer's window.Elements.
   * (scenery-internal)
   */
  public setAttributeToElement( attribute: string, attributeValue: IntentionalAny, options?: IntentionalAny ): void {

    // eslint-disable-next-line phet/bad-typescript-text
    options = merge( {
      // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
      // for setting certain attributes (e.g. MathML).
      namespace: null,

      // set as a javascript property instead of an attribute on the DOM Element.
      type: 'attribute',

      elementName: PRIMARY_SIBLING, // see this.getElementName() for valid values, default to the primary sibling

      // {HTMLElement|null} - element that will directly receive the input rather than looking up by name, if
      // provided, elementName option will have no effect
      element: null
    }, options );

    // There may not be an element due to order of operations, or if there is no default primary sibling.
    const element = options.element || this.getElementByName( options.elementName );
    if ( !element ) {
      return;
    }

    // For dynamic strings, we may need to retrieve the actual value.
    const rawAttributeValue = PDOMUtils.unwrapProperty( attributeValue );

    // remove directional formatting that may surround strings if they are translatable
    let attributeValueWithoutMarks = rawAttributeValue;
    if ( typeof rawAttributeValue === 'string' ) {
      attributeValueWithoutMarks = stripEmbeddingMarks( rawAttributeValue );
    }

    if ( attribute === DISABLED_ATTRIBUTE_NAME && !this.display.interactive ) {

      // The presence of the `disabled` attribute means it is always disabled.
      this._preservedDisabledValue = options.type === 'property' ? attributeValueWithoutMarks : true;
    }

    if ( options.namespace ) {
      element.setAttributeNS( options.namespace, attribute, attributeValueWithoutMarks );
    }
    else if ( options.type === 'property' ) {
      element[ attribute ] = attributeValueWithoutMarks;
    }
    else {
      element.setAttribute( attribute, attributeValueWithoutMarks );
    }
  }

  /**
   * Remove attribute from one of the peer's window.Elements.
   * (scenery-internal)
   */
  public removeAttributeFromElement( attribute: string, options?: IntentionalAny ): void {

    // eslint-disable-next-line phet/bad-typescript-text
    options = merge( {
      // {string|null} - If non-null, will set the attribute with the specified namespace. This can be required
      // for setting certain attributes (e.g. MathML).
      namespace: null,

      elementName: PRIMARY_SIBLING, // see this.getElementName() for valid values, default to the primary sibling

      // {HTMLElement|null} - element that will directly receive the input rather than looking up by name, if
      // provided, elementName option will have no effect
      element: null
    }, options );

    const element = options.element || this.getElementByName( options.elementName );

    if ( options.namespace ) {
      element.removeAttributeNS( options.namespace, attribute );
    }
    else if ( attribute === DISABLED_ATTRIBUTE_NAME && !this.display.interactive ) {
      // maintain our interal disabled state in case the display toggles back to be interactive.
      this._preservedDisabledValue = false;
    }
    else {
      element.removeAttribute( attribute );
    }
  }

  /**
   * Remove the given attribute from all peer elements
   * (scenery-internal)
   */
  public removeAttributeFromAllElements( attribute: string ): void {
    assert && assert( attribute !== DISABLED_ATTRIBUTE_NAME, 'this method does not currently support disabled, to make Display.interactive toggling easier to implement' );
    this._primarySibling && this._primarySibling.removeAttribute( attribute );
    this._labelSibling && this._labelSibling.removeAttribute( attribute );
    this._descriptionSibling && this._descriptionSibling.removeAttribute( attribute );
    this._accessibleParagraphSibling && this._accessibleParagraphSibling.removeAttribute( attribute );
    this._containerParent && this._containerParent.removeAttribute( attribute );
  }

  /**
   * Add the provided className to the element's classList.
   */
  public setClassToElement( className: string, options: IntentionalAny ): void {

    // eslint-disable-next-line phet/bad-typescript-text
    options = merge( {

      // Name of the element who we are adding the class to, see this.getElementName() for valid values
      elementName: PRIMARY_SIBLING
    }, options );

    this.getElementByName( options.elementName ).classList.add( className );
  }

  /**
   * Remove the specified className from the element.
   */
  public removeClassFromElement( className: string, options: IntentionalAny ): void {

    // eslint-disable-next-line phet/bad-typescript-text
    options = merge( {

      // Name of the element who we are removing the class from, see this.getElementName() for valid values
      elementName: PRIMARY_SIBLING
    }, options );

    this.getElementByName( options.elementName ).classList.remove( className );
  }

  /**
   * Set either association attribute (aria-labelledby/describedby) on one of this peer's Elements
   * (scenery-internal)
   * @param attribute - either aria-labelledby or aria-describedby
   * @param associationObject - see addAriaLabelledbyAssociation() for schema
   */
  public setAssociationAttribute( attribute: string, associationObject: IntentionalAny ): void {
    assert && assert( PDOMUtils.ASSOCIATION_ATTRIBUTES.includes( attribute ),
      `unsupported attribute for setting with association object: ${attribute}` );

    const otherNodePDOMInstances = associationObject.otherNode.getPDOMInstances();

    // If the other node hasn't been added to the scene graph yet, it won't have any accessible instances, so no op.
    // This will be recalculated when that node is added to the scene graph
    if ( otherNodePDOMInstances.length > 0 ) {

      // We are just using the first PDOMInstance for simplicity, but it is OK because the accessible
      // content for all PDOMInstances will be the same, so the Accessible Names (in the browser's
      // accessibility tree) of elements that are referenced by the attribute value id will all have the same content
      const firstPDOMInstance = otherNodePDOMInstances[ 0 ];

      // Handle a case where you are associating to yourself, and the peer has not been constructed yet.
      if ( firstPDOMInstance === this.pdomInstance ) {
        firstPDOMInstance.peer = this;
      }

      assert && assert( firstPDOMInstance.peer, 'peer should exist' );

      // we can use the same element's id to update all of this Node's peers
      const otherPeerElement = firstPDOMInstance.peer.getElementByName( associationObject.otherElementName );

      const element = this.getElementByName( associationObject.thisElementName );

      // to support any option order, no-op if the peer element has not been created yet.
      if ( element && otherPeerElement ) {

        // only update associations if the requested peer element has been created
        // NOTE: in the future, we would like to verify that the association exists but can't do that yet because
        // we have to support cases where we set label association prior to setting the sibling/parent tagName
        const previousAttributeValue = element.getAttribute( attribute ) || '';

        const newAttributeValue = [ previousAttributeValue.trim(), otherPeerElement.id ].join( ' ' ).trim();

        // add the id from the new association to the value of the HTMLElement's attribute.
        this.setAttributeToElement( attribute, newAttributeValue, {
          elementName: associationObject.thisElementName
        } );
      }
    }
  }

  /**
   * The contentElement will either be a label or description element. The contentElement will be sorted relative to
   * the primarySibling. Its placement will also depend on whether or not this node wants to append this element,
   * see setAppendLabel() and setAppendDescription(). By default, the "content" element will be placed before the
   * primarySibling.
   *
   * NOTE: This function assumes it is called on label sibling before description sibling for inserting elements
   * into the correct order.
   */
  private arrangeContentElement( contentElement: HTMLElement, appendElement: boolean ): void {

    // if there is a containerParent
    if ( this.topLevelElements[ 0 ] === this._containerParent ) {
      assert && assert( this.topLevelElements.length === 1 );

      if ( appendElement ) {
        this._containerParent.appendChild( contentElement );
      }
      else {
        this._containerParent.insertBefore( contentElement, this._primarySibling! );
      }
    }

    // If there are multiple top level nodes
    else {
      if ( assert && !appendElement ) {
        assert && assert( this._primarySibling, 'There must be a primary sibling to sort relative to it if not appending.' );
      }

      // keep this.topLevelElements in sync
      arrayRemove( this.topLevelElements, contentElement );
      const indexOfPrimarySibling = this.topLevelElements.indexOf( this._primarySibling! );

      // if appending, just insert at at end of the top level elements
      const insertIndex = appendElement ? this.topLevelElements.length : indexOfPrimarySibling;
      this.topLevelElements.splice( insertIndex, 0, contentElement );
    }
  }

  /**
   * Is this peer hidden in the PDOM
   */
  public isVisible(): boolean {
    if ( assert ) {

      let visibleElements = 0;
      this.topLevelElements.forEach( element => {

        // support property or attribute
        if ( !element.hidden && !element.hasAttribute( 'hidden' ) ) {
          visibleElements += 1;
        }
      } );
      assert( this.visible ? visibleElements === this.topLevelElements.length : visibleElements === 0,
        'some of the peer\'s elements are visible and some are not' );

    }
    return this.visible === null ? true : this.visible; // default to true if visibility hasn't been set yet.
  }

  /**
   * Set whether the peer is visible in the PDOM
   */
  public setVisible( visible: boolean ): void {
    if ( this.visible !== visible ) {

      this.visible = visible;
      for ( let i = 0; i < this.topLevelElements.length; i++ ) {
        const element = this.topLevelElements[ i ];
        if ( visible ) {
          this.removeAttributeFromElement( 'hidden', { element: element } );
        }
        else {
          this.setAttributeToElement( 'hidden', '', { element: element } );
        }
      }
    }
  }

  /**
   * Returns if this peer is focused. A peer is focused if its primarySibling is focused.
   * (scenery-internal)
   */
  public isFocused(): boolean | null {
    const visualFocusTrail = guessVisualTrail( this.trail, this.display.rootNode );

    return pdomFocusProperty.value && pdomFocusProperty.value.trail.equals( visualFocusTrail );
  }

  /**
   * Focus the primary sibling. If this peer is not visible, this is a no-op (native behavior).
   * (scenery-internal)
   */
  public focus(): void {
    assert && assert( this._primarySibling, 'must have a primary sibling to focus' );

    // We do not want to steal focus from any parent application. For example, if this element is in an iframe.
    // See https://github.com/phetsims/joist/issues/897.
    if ( FocusManager.windowHasFocusProperty.value ) {
      this._primarySibling!.focus();
    }
  }

  /**
   * Blur the primary sibling.
   * (scenery-internal)
   */
  public blur(): void {
    assert && assert( this._primarySibling, 'must have a primary sibling to blur' );

    // no op by the browser if primary sibling does not have focus
    this._primarySibling!.blur();
  }

  /**
   * Make the peer focusable. Only the primary sibling is ever considered focusable.
   */
  public setFocusable( focusable: boolean ): void {

    const peerHadFocus = this.isFocused();
    if ( this.focusable !== focusable && this.primarySibling ) {
      this.focusable = focusable;
      PDOMUtils.overrideFocusWithTabIndex( this.primarySibling, focusable );

      // in Chrome, if tabindex is removed and the element is not focusable by default the element is blurred.
      // This behavior is reasonable, and we want to enforce it in other browsers for consistency. See
      // https://github.com/phetsims/scenery/issues/967
      if ( peerHadFocus && !focusable ) {
        this.blur();
      }
    }
  }

  /**
   * Sets the heading content
   * (scenery-internal)
   * @param content --- NOTE that this is called from update(), where the value gets string-ified, so this
   * type info is correct
   */
  public setHeadingContent( content: string | null ): void {
    assert && assert( content === null || typeof content === 'string', 'incorrect heading content type' );

    // no-op to support any option order
    if ( !this._headingSibling ) {
      return;
    }

    PDOMUtils.setTextContent( this._headingSibling, content );
  }

  /**
   * Responsible for setting the content for the label sibling
   * (scenery-internal)
   * content - the content for the label sibling.
   */
  public setLabelSiblingContent( content: string | null ): void {
    assert && assert( content === null || typeof content === 'string', 'incorrect label content type' );

    // no-op to support any option order
    if ( !this._labelSibling ) {
      return;
    }

    PDOMUtils.setTextContent( this._labelSibling, content );
  }

  /**
   * Responsible for setting the content for the description sibling
   * (scenery-internal)
   * @param content - the content for the description sibling.
   */
  public setDescriptionSiblingContent( content: string | null ): void {
    assert && assert( content === null || typeof content === 'string', 'incorrect description content type' );

    // no-op to support any option order
    if ( !this._descriptionSibling ) {
      return;
    }
    PDOMUtils.setTextContent( this._descriptionSibling, content );
  }

  /**
   * Responsible for setting the content for the paragraph sibling.
   * (scenery-internal)
   * @param content - the content for the paragraph sibling.
   */
  public setAccessibleParagraphContent( content: string | null ): void {
    assert && assert( content === null || typeof content === 'string', 'incorrect description content type' );

    // no-op to support any option order
    if ( !this._accessibleParagraphSibling ) {
      return;
    }
    PDOMUtils.setTextContent( this._accessibleParagraphSibling, content );
  }

  /**
   * Responsible for setting the content for the primary sibling
   * (scenery-internal)
   * @param content - the content for the primary sibling.
   */
  public setPrimarySiblingContent( content: string | null ): void {

    // no-op to support any option order
    if ( !this._primarySibling ) {
      return;
    }

    assert && assert( content === null || typeof content === 'string', 'incorrect inner content type' );
    assert && assert( this.pdomInstance.children.length === 0, 'descendants exist with accessible content, innerContent cannot be used' );
    assert && assert( PDOMUtils.tagNameSupportsContent( this._primarySibling.tagName ),
      `tagName: ${this.node!.tagName} does not support inner content` );

    PDOMUtils.setTextContent( this._primarySibling, content );
  }

  private getElementId( siblingName: string, stringId: string ): string {
    return `display${this.display.id}-${siblingName}-${stringId}`;
  }

  /**
   * Set ids on elements so for easy lookup with document.getElementById. Also assign a unique
   * data attribute to the elements so that scenery can look up an element from a Trail (mostly
   * for input handling).
   *
   * Note that dataset isn't supported by all namespaces (like MathML) so we need to use setAttribute.
   */
  public updateIndicesStringAndElementIds(): void {
    const indices = this.pdomInstance.getPDOMInstanceUniqueId();

    if ( this._primarySibling ) {
      this._primarySibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._primarySibling.id = this.getElementId( 'primary', indices );
    }
    if ( this._labelSibling ) {
      this._labelSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._labelSibling.id = this.getElementId( 'label', indices );
    }
    if ( this._descriptionSibling ) {
      this._descriptionSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._descriptionSibling.id = this.getElementId( 'description', indices );
    }
    if ( this._headingSibling ) {
      this._headingSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._headingSibling.id = this.getElementId( 'heading', indices );
    }
    if ( this._accessibleParagraphSibling ) {
      this._accessibleParagraphSibling.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._accessibleParagraphSibling.id = this.getElementId( 'paragraph', indices );
    }
    if ( this._containerParent ) {
      this._containerParent.setAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID, indices );
      this._containerParent.id = this.getElementId( 'container', indices );
    }
  }

  /**
   * Recursively set this PDOMPeer and children to be disabled. This will overwrite any previous value of disabled
   * that may have been set, but will keep track of the old value, and restore its state upon re-enabling.
   */
  public recursiveDisable( disabled: boolean ): void {

    if ( this._primarySibling ) {
      if ( disabled ) {
        // @ts-expect-error
        this._preservedDisabledValue = this._primarySibling.disabled;
        // @ts-expect-error
        this._primarySibling.disabled = true;
      }
      else {
        // @ts-expect-error
        this._primarySibling.disabled = this._preservedDisabledValue;
      }
    }

    for ( let i = 0; i < this.pdomInstance.children.length; i++ ) {
      this.pdomInstance.children[ i ].peer!.recursiveDisable( disabled );
    }
  }

  /**
   * Removes external references from this peer, and places it in the pool.
   * (scenery-internal)
   */
  public dispose(): void {
    this.isDisposed = true;

    // remove focus if the disposed peer is the active element
    if ( this._primarySibling ) {
      this.blur();

      // @ts-expect-error
      this._primarySibling.removeEventListener( 'blur', this.blurEventListener );
      // @ts-expect-error
      this._primarySibling.removeEventListener( 'focus', this.focusEventListener );
    }

    // zero-out references
    // @ts-expect-error
    this.pdomInstance = null;
    this.node = null;
    // @ts-expect-error
    this.display = null;
    // @ts-expect-error
    this.trail = null;
    // @ts-expect-error
    this._primarySibling = null;
    this._labelSibling = null;
    this._descriptionSibling = null;
    this._headingSibling = null;
    this._accessibleParagraphSibling = null;
    this._containerParent = null;
    this.focusable = null;

    // for now
    // @ts-expect-error
    this.freeToPool();
  }

  // specifies valid associations between related PDOMPeers in the DOM
  public static readonly PRIMARY_SIBLING = PRIMARY_SIBLING; // associate with all accessible content related to this peer
  public static readonly HEADING_SIBLING = HEADING_SIBLING; // associate with just the heading content of this peer
  public static readonly LABEL_SIBLING = LABEL_SIBLING; // associate with just the label content of this peer
  public static readonly DESCRIPTION_SIBLING = DESCRIPTION_SIBLING; // associate with just the description content of this peer
  public static readonly PARAGRAPH_SIBLING = ACCESSIBLE_PARAGRAPH_SIBLING; // associate with just the paragraph content of this peer
  public static readonly CONTAINER_PARENT = CONTAINER_PARENT; // associate with everything under the container parent of this peer
}

scenery.register( 'PDOMPeer', PDOMPeer );

// Set up pooling
// @ts-expect-error
Poolable.mixInto( PDOMPeer, {
  initialize: PDOMPeer.prototype.initializePDOMPeer
} );

//--------------------------------------------------------------------------
// Helper functions
//--------------------------------------------------------------------------

/**
 * Create a sibling element for the PDOMPeer.
 * TODO: this should be inlined with the PDOMUtils method https://github.com/phetsims/scenery/issues/1581
 * @param tagName
 * @param focusable
 * @param [options] - passed along to PDOMUtils.createElement
 */
function createElement( tagName: string, focusable: boolean, options?: IntentionalAny ): HTMLElement {
  // eslint-disable-next-line phet/bad-typescript-text
  options = merge( {

    // {string|null} - addition to the trailId, separated by a hyphen to identify the different siblings within
    // the document
    siblingName: null,

    // {boolean} - if true, DOM input events received on the element will not be dispatched as SceneryEvents in Input.js
    // see ParallelDOM.setExcludeLabelSiblingFromInput for more information
    excludeFromInput: false
  }, options );

  const newElement = PDOMUtils.createElement( tagName, focusable, options );

  if ( options.excludeFromInput ) {
    // @ts-expect-error -- only accepts string?
    newElement.setAttribute( PDOMUtils.DATA_EXCLUDE_FROM_INPUT, true );
  }

  return newElement;
}

export default PDOMPeer;